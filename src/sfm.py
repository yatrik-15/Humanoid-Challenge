"""Joint structure-from-motion over the frames of both recordings.

hloc extracts DISK features and matches them with LightGlue; pycolmap runs incremental
mapping with one shared camera. Frames of A and B are matched against each other, so both
recordings end up in a single coordinate frame and no later registration step is needed.

pycolmap's Image.cam_from_world is T_cw (world -> camera).

Writes sfm/model, sfm/undistorted (PINHOLE), sfm/hloc (features, matches),
       sfm/sfm_report.json, sfm/poses.json, sfm/trajectories.png
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pycolmap
from hloc import (
    extract_features,
    match_features,
    pairs_from_exhaustive,
    reconstruction,
)


def get_feature_conf(feature_type: str, max_keypoints: int) -> dict:
    """Get hloc feature extraction configuration.

    Args:
        feature_type: "superpoint" or "disk".
        max_keypoints: Maximum keypoints per image.

    Returns:
        hloc feature config dict.
    """
    if feature_type == "superpoint":
        conf = extract_features.confs["superpoint_aachen"].copy()
        conf["model"] = conf["model"].copy()
        conf["model"]["max_keypoints"] = max_keypoints
    elif feature_type == "disk":
        conf = extract_features.confs["disk"].copy()
        conf["model"] = conf["model"].copy()
        conf["model"]["max_keypoints"] = max_keypoints
    else:
        raise ValueError(f"Unknown feature type: {feature_type}. Use 'superpoint' or 'disk'.")
    return conf


def get_matcher_conf(feature_type: str, matcher_type: str) -> dict:
    """Get hloc matcher configuration.

    Args:
        feature_type: "superpoint" or "disk".
        matcher_type: "lightglue" or "superglue".

    Returns:
        hloc matcher config dict.
    """
    if matcher_type == "lightglue":
        key = f"{feature_type}+lightglue"
    elif matcher_type == "superglue":
        key = "superglue"
    else:
        raise ValueError(f"Unknown matcher type: {matcher_type}")

    if key not in match_features.confs:
        raise ValueError(f"Matcher config '{key}' not found in hloc.")

    return match_features.confs[key].copy()


def run_joint_sfm(out_dir: Path, config: dict) -> pycolmap.Reconstruction:
    """Run joint SfM on extracted frames from both videos.

    Args:
        out_dir: Output directory (must contain out_dir/frames/ with extracted frames).
        config: SfM config dict from config.yaml.

    Returns:
        pycolmap.Reconstruction object.
    """
    frames_dir = out_dir / "frames"
    sfm_dir = out_dir / "sfm"
    hloc_dir = sfm_dir / "hloc"
    sfm_dir.mkdir(parents=True, exist_ok=True)
    hloc_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest to get image list
    manifest_path = frames_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    image_list = [entry["frame_name"] for entry in manifest]
    a_images = [e["frame_name"] for e in manifest if e["epoch"] == "A"]
    b_images = [e["frame_name"] for e in manifest if e["epoch"] == "B"]

    print(f"  Images: {len(image_list)} total ({len(a_images)} A, {len(b_images)} B)")

    # --- Feature Extraction ---
    print("\n  [1/4] Extracting features...")
    feature_conf = get_feature_conf(
        config["feature_type"],
        config["max_keypoints"],
    )
    feature_path = extract_features.main(
        conf=feature_conf,
        image_dir=frames_dir,
        export_dir=hloc_dir,
        image_list=image_list,
        overwrite=False,
    )
    print(f"  Features saved to: {feature_path}")

    # --- Generate Pairs ---
    print("\n  [2/4] Generating match pairs...")
    pairs_path = hloc_dir / "pairs.txt"

    if config["match_pairs"] == "exhaustive":
        pairs_from_exhaustive.main(
            output=pairs_path,
            image_list=image_list,
        )
    else:
        # Retrieval-based pairing (uses NetVLAD)
        from hloc import pairs_from_retrieval

        retrieval_conf = extract_features.confs["netvlad"]
        retrieval_path = extract_features.main(
            conf=retrieval_conf,
            image_dir=frames_dir,
            export_dir=hloc_dir,
            image_list=image_list,
            overwrite=False,
        )
        pairs_from_retrieval.main(
            descriptors=retrieval_path,
            output=pairs_path,
            num_matched=config["retrieval_top_k"],
        )

    # Count pairs
    with open(pairs_path) as f:
        n_pairs = sum(1 for _ in f)
    print(f"  Generated {n_pairs} pairs")

    # --- Feature Matching ---
    print("\n  [3/4] Matching features...")
    matcher_conf = get_matcher_conf(
        config["feature_type"],
        config["matcher_type"],
    )
    # hloc needs an explicit matches path when `features` is given as a Path
    matches_path = match_features.main(
        conf=matcher_conf,
        pairs=pairs_path,
        features=feature_path,
        matches=hloc_dir / f"{matcher_conf['output']}.h5",
        overwrite=False,
    )
    print(f"  Matches saved to: {matches_path}")

    # --- Reconstruction ---
    print("\n  [4/4] Running SfM reconstruction...")

    # one shared camera: both videos come from the same phone
    camera_mode = pycolmap.CameraMode.SINGLE

    # keys are pycolmap.IncrementalPipelineOptions attribute names
    mapper_options = dict(config.get("mapper", {}))

    model = reconstruction.main(
        sfm_dir=sfm_dir / "model",
        image_dir=frames_dir,
        pairs=pairs_path,
        features=feature_path,
        matches=matches_path,
        camera_mode=camera_mode,
        verbose=True,
        min_match_score=config.get("min_match_score"),
        image_list=image_list,  # avoids importing manifest.csv/json as images
        mapper_options=mapper_options or None,
    )
    if model is None:
        raise RuntimeError(
            "Joint SfM produced no model. Check overlap/texture between the videos; "
            f"COLMAP logs are in {sfm_dir / 'model'}."
        )

    # --- Report ---
    reg_image_ids = model.reg_image_ids()
    reg_images = [model.image(img_id) for img_id in reg_image_ids]
    reg_names = [img.name for img in reg_images]

    reg_a = [n for n in reg_names if n.startswith("A_")]
    reg_b = [n for n in reg_names if n.startswith("B_")]

    mean_reproj_error = model.compute_mean_reprojection_error()
    num_points3d = model.num_points3D()
    mean_track_length = model.compute_mean_track_length()

    report = {
        "total_images": len(image_list),
        "registered_images": len(reg_image_ids),
        "registered_A": len(reg_a),
        "registered_B": len(reg_b),
        "unregistered_A": len(a_images) - len(reg_a),
        "unregistered_B": len(b_images) - len(reg_b),
        "num_points3D": num_points3d,
        "mean_reprojection_error": round(mean_reproj_error, 4),
        "mean_track_length": round(mean_track_length, 2),
        "num_cameras": model.num_cameras(),
    }

    # Save camera intrinsics
    for cam_id in model.cameras:
        cam = model.camera(cam_id)
        report[f"camera_{cam_id}"] = {
            "model": cam.model_name,
            "width": cam.width,
            "height": cam.height,
            "params": cam.params.tolist(),
        }

    report_path = sfm_dir / "sfm_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n  === SfM Report ===")
    print(
        f"  Registered: {report['registered_images']}/{report['total_images']} "
        f"(A: {report['registered_A']}/{len(a_images)}, "
        f"B: {report['registered_B']}/{len(b_images)})"
    )
    print(f"  3D Points: {num_points3d}")
    print(f"  Mean reprojection error: {mean_reproj_error:.4f} px")
    print(f"  Mean track length: {mean_track_length:.2f}")
    print(f"  Report saved to: {report_path}")

    # Check acceptance criteria
    a_ratio = len(reg_a) / max(len(a_images), 1)
    b_ratio = len(reg_b) / max(len(b_images), 1)

    if a_ratio < 0.5 or b_ratio < 0.5:
        print(
            f"\n  WARNING: Registration ratio too low "
            f"(A: {a_ratio:.0%}, B: {b_ratio:.0%}). "
            f"Joint SfM may have failed to register both videos."
        )
    if mean_reproj_error > 2.0:
        print(f"\n  WARNING: Mean reprojection error is high ({mean_reproj_error:.2f} px)")

    undist_dir = undistort(sfm_dir, frames_dir, reg_names)
    poses = export_poses(pycolmap.Reconstruction(undist_dir / "sparse"), sfm_dir / "poses.json")
    plot_trajectories(poses, sfm_dir / "trajectories.png")
    return model


def undistort(sfm_dir: Path, frames_dir: Path, image_names: list[str]) -> Path:
    """Undistort registered images to a PINHOLE model (pycolmap.undistort_images).

    Writes <sfm_dir>/undistorted/{images/, sparse/}. All downstream stages (depth,
    fusion, diff) use these images and this pinhole model, never the distorted ones.
    """
    undist_dir = sfm_dir / "undistorted"
    if undist_dir.exists():
        shutil.rmtree(undist_dir)
    pycolmap.undistort_images(
        output_path=undist_dir,
        input_path=sfm_dir / "model",
        image_path=frames_dir,
        image_names=image_names,
        output_type="COLMAP",
    )
    return undist_dir


def export_poses(model: pycolmap.Reconstruction, out_path: Path) -> dict[str, dict]:
    """Export per-image pose and pinhole intrinsics to JSON.

    Each entry: {"epoch": "A"|"B", "T_cw": 4x4 world->camera (OpenCV camera frame),
                 "K": 3x3, "width": int, "height": int}.
    """
    poses: dict[str, dict] = {}
    for img_id in model.reg_image_ids():
        img = model.image(img_id)
        cam = model.camera(img.camera_id)
        if cam.model_name not in ("PINHOLE", "SIMPLE_PINHOLE"):
            raise ValueError(f"Expected undistorted pinhole camera, got {cam.model_name}")
        T_cw = np.eye(4)
        T_cw[:3, :] = np.asarray(img.cam_from_world().matrix())
        poses[img.name] = {
            "epoch": img.name.split("_")[0],
            "T_cw": T_cw.tolist(),
            "K": np.asarray(cam.calibration_matrix()).tolist(),
            "width": int(cam.width),
            "height": int(cam.height),
        }
    with open(out_path, "w") as f:
        json.dump(dict(sorted(poses.items())), f, indent=1)
    print(f"  Poses ({len(poses)}) saved to: {out_path}")
    return poses


def plot_trajectories(poses: dict[str, dict], out_path: Path) -> None:
    """Save top-down (x-y) and side (x-z) plots of A and B camera centres in one frame."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.utils.geometry import camera_center

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for epoch, color in (("A", "tab:blue"), ("B", "tab:orange")):
        names = sorted(n for n, p in poses.items() if p["epoch"] == epoch)
        if not names:
            continue
        C = np.array([camera_center(np.array(poses[n]["T_cw"])) for n in names])
        for ax, (i, j) in zip(axes, ((0, 1), (0, 2))):
            ax.plot(C[:, i], C[:, j], "-o", ms=3, color=color, label=f"{epoch} ({len(names)})")
    for ax, title, yl in zip(axes, ("top-down (x-y)", "side (x-z)"), ("y", "z")):
        ax.set_title(f"Camera centres, SfM frame: {title}")
        ax.set_xlabel("x")
        ax.set_ylabel(yl)
        ax.axis("equal")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Trajectory plot saved to: {out_path}")
