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

import cv2
import numpy as np
import pycolmap
from hloc import (
    extract_features,
    match_features,
    pairs_from_exhaustive,
    reconstruction,
)

from src.pairs import windowed_pairs


def feature_conf(feature_type: str, max_keypoints: int) -> dict:
    """hloc feature extractor config ("disk" or "superpoint") with a keypoint budget."""
    names = {"disk": "disk", "superpoint": "superpoint_aachen"}
    if feature_type not in names:
        raise ValueError(f"Unknown feature type {feature_type!r}; use 'disk' or 'superpoint'.")
    conf = extract_features.confs[names[feature_type]].copy()
    conf["model"] = {**conf["model"], "max_keypoints": max_keypoints}
    return conf


def matcher_conf(feature_type: str, matcher_type: str) -> dict:
    """hloc matcher config: LightGlue for the given features, or SuperGlue."""
    if matcher_type not in ("lightglue", "superglue"):
        raise ValueError(f"Unknown matcher type {matcher_type!r}")
    key = f"{feature_type}+lightglue" if matcher_type == "lightglue" else "superglue"
    if key not in match_features.confs:
        raise ValueError(f"Matcher config {key!r} not found in hloc.")
    return match_features.confs[key].copy()


def write_pairs(config: dict, image_list: list[str], frames_dir: Path, hloc_dir: Path) -> Path:
    """Write the image pairs to match (sfm.match_pairs) and return the pairs file."""
    pairs_path = hloc_dir / "pairs.txt"
    mode = config["match_pairs"]
    if mode == "exhaustive":
        pairs_from_exhaustive.main(output=pairs_path, image_list=image_list)
    elif mode == "windowed":
        pairs = windowed_pairs(image_list, config["pair_window"], config["pair_cross_stride"])
        pairs_path.write_text("".join(f"{a} {b}\n" for a, b in pairs))
    elif mode == "retrieval":
        from hloc import pairs_from_retrieval

        descriptors = extract_features.main(
            conf=extract_features.confs["netvlad"],
            image_dir=frames_dir,
            export_dir=hloc_dir,
            image_list=image_list,
            overwrite=False,
        )
        pairs_from_retrieval.main(
            descriptors=descriptors, output=pairs_path, num_matched=config["retrieval_top_k"]
        )
    else:
        raise ValueError(f"Unknown match_pairs {mode!r}")
    with open(pairs_path) as f:
        print(f"  Generated {sum(1 for _ in f)} pairs")
    return pairs_path


def camera_options(config: dict, first_image: Path) -> tuple[dict, dict]:
    """(image_options, mapper_options) for hloc's reconstruction.

    Self-calibration can trade a too-long focal length against strong radial distortion
    when the tracked points sit at similar depths, which flattens all depth. With
    `known_focal_rel` the lens is set as the prior and kept fixed; distortion is still refined.
    """
    image_options = {"camera_model": config["camera_model"]}
    mapper_options = dict(config.get("mapper", {}))  # pycolmap.IncrementalPipelineOptions fields
    f_rel = config.get("known_focal_rel")
    if f_rel:
        if config["camera_model"] != "SIMPLE_RADIAL":
            raise ValueError("known_focal_rel is implemented for SIMPLE_RADIAL (f, cx, cy, k)")
        h, w = cv2.imread(str(first_image)).shape[:2]
        image_options["camera_params"] = f"{f_rel * w},{w / 2},{h / 2},0"
        mapper_options["ba_refine_focal_length"] = False
        print(f"  Known lens: f = {f_rel} x width = {f_rel * w:.1f} px, kept fixed")
    return image_options, mapper_options


def sfm_report(model: pycolmap.Reconstruction, image_list: list[str], max_abs_radial: float) -> dict:
    """Registration counts per recording, reprojection error, track length and intrinsics,
    with a warning when the solved distortion is implausible for a phone's main camera."""
    reg_names = [model.image(i).name for i in model.reg_image_ids()]
    total = {e: sum(n.startswith(f"{e}_") for n in image_list) for e in ("A", "B")}
    reg = {e: sum(n.startswith(f"{e}_") for n in reg_names) for e in ("A", "B")}
    report = {
        "total_images": len(image_list),
        "registered_images": len(reg_names),
        "registered_A": reg["A"],
        "registered_B": reg["B"],
        "unregistered_A": total["A"] - reg["A"],
        "unregistered_B": total["B"] - reg["B"],
        "num_points3D": model.num_points3D(),
        "mean_reprojection_error": round(model.compute_mean_reprojection_error(), 4),
        "mean_track_length": round(model.compute_mean_track_length(), 2),
        "num_cameras": model.num_cameras(),
    }
    for cam_id in model.cameras:
        cam = model.camera(cam_id)
        report[f"camera_{cam_id}"] = {
            "model": cam.model_name,
            "width": cam.width,
            "height": cam.height,
            "params": cam.params.tolist(),
        }
    cam = model.camera(next(iter(model.cameras)))
    if cam.model_name == "SIMPLE_RADIAL":
        k = float(cam.params[3])
        report["radial_distortion"] = round(k, 4)
        report["intrinsics_suspicious"] = abs(k) > max_abs_radial
        if report["intrinsics_suspicious"]:
            print(
                f"  WARNING: radial distortion {k:.3f} is implausible for a phone's main camera; "
                "self-calibration probably traded it against the focal length, which flattens "
                "depth. Re-run with --known_focal_rel (see README)."
            )
    print(
        f"  Registered: {report['registered_images']}/{report['total_images']} "
        f"(A: {reg['A']}/{total['A']}, B: {reg['B']}/{total['B']})"
    )
    print(f"  3D points: {report['num_points3D']}, mean track length {report['mean_track_length']:.2f}")
    print(f"  Mean reprojection error: {report['mean_reprojection_error']:.4f} px")
    if min(reg[e] / max(total[e], 1) for e in ("A", "B")) < 0.5:
        print("  WARNING: under half of one recording registered; the joint model may not link A and B.")
    if report["mean_reprojection_error"] > 2.0:
        print("  WARNING: mean reprojection error above 2 px.")
    return report


def run_joint_sfm(out_dir: Path, config: dict) -> pycolmap.Reconstruction:
    """Stage entry: features, pairs, matches and one incremental reconstruction of the
    frames of both recordings, then the undistorted model, poses and plots."""
    frames_dir, sfm_dir = out_dir / "frames", out_dir / "sfm"
    hloc_dir = sfm_dir / "hloc"
    hloc_dir.mkdir(parents=True, exist_ok=True)
    with open(frames_dir / "manifest.json") as f:
        image_list = [entry["frame_name"] for entry in json.load(f)]
    n_a = sum(n.startswith("A_") for n in image_list)
    print(f"  Images: {len(image_list)} total ({n_a} A, {len(image_list) - n_a} B)")

    print("\n  [1/4] Extracting features...")
    features = extract_features.main(
        conf=feature_conf(config["feature_type"], config["max_keypoints"]),
        image_dir=frames_dir,
        export_dir=hloc_dir,
        image_list=image_list,
        overwrite=False,
    )
    print("\n  [2/4] Generating match pairs...")
    pairs = write_pairs(config, image_list, frames_dir, hloc_dir)
    print("\n  [3/4] Matching features...")
    m_conf = matcher_conf(config["feature_type"], config["matcher_type"])
    matches = match_features.main(  # hloc needs an explicit matches path when features is a Path
        conf=m_conf,
        pairs=pairs,
        features=features,
        matches=hloc_dir / f"{m_conf['output']}.h5",
        overwrite=False,
    )
    print("\n  [4/4] Running SfM reconstruction...")
    image_options, mapper_options = camera_options(config, frames_dir / image_list[0])
    model = reconstruction.main(
        sfm_dir=sfm_dir / "model",
        image_dir=frames_dir,
        pairs=pairs,
        features=features,
        matches=matches,
        camera_mode=pycolmap.CameraMode.SINGLE,  # one shared camera: both videos from one phone
        verbose=True,
        min_match_score=config.get("min_match_score"),
        image_list=image_list,  # avoids importing manifest.csv/json as images
        image_options=image_options,
        mapper_options=mapper_options or None,
    )
    if model is None:
        raise RuntimeError(
            "Joint SfM produced no model. Check overlap/texture between the videos; "
            f"COLMAP logs are in {sfm_dir / 'model'}."
        )
    report = sfm_report(model, image_list, config["max_abs_radial"])
    with open(sfm_dir / "sfm_report.json", "w") as f:
        json.dump(report, f, indent=2)

    reg_names = [model.image(i).name for i in model.reg_image_ids()]
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
