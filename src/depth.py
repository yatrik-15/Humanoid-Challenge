"""Monocular depth per frame, aligned to the sparse SfM depth.

Depth Anything V2 predicts relative inverse depth. `depth_align.fit_inverse_depth_affine`
fits a scale and shift per frame against that frame's triangulated points (RANSAC); frames
with a poor fit are dropped rather than fused.

Reads  align/sparse, sfm/undistorted/images/
Writes depth/<stem>.npy           float32 z-depth in world units, 0 = invalid (accepted frames only)
       depth/depth_report.json    per-frame fit
       depth/alignment_plots.png  aligned vs. sparse depth for a few frames
"""

import json
from pathlib import Path

import numpy as np
import pycolmap

from src.depth_align import DepthFit, apply_fit, fit_inverse_depth_affine
from src.utils.geometry import pixel_index, to_homogeneous_4x4, transform_points


def sparse_depth_samples(model: pycolmap.Reconstruction, image_id: int) -> tuple[np.ndarray, np.ndarray]:
    """COLMAP image coordinates (N, 2) and camera z-depth (N,) of an image's triangulated points."""
    img = model.image(image_id)
    T_cw = to_homogeneous_4x4(np.asarray(img.cam_from_world().matrix()))
    uv, X = [], []
    for p2d in img.points2D:
        if p2d.has_point3D():
            uv.append(p2d.xy)
            X.append(model.point3D(p2d.point3D_id).xyz)
    if not uv:
        return np.zeros((0, 2)), np.zeros(0)
    z = transform_points(T_cw, np.array(X))[:, 2]
    return np.array(uv, dtype=np.float64), z


class DepthModel:
    """Thin wrapper around a Hugging Face Depth Anything V2 checkpoint."""

    def __init__(self, model_id: str, device: str):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.torch = torch
        self.device = device
        self.proc = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """(H, W, 3) uint8 RGB -> (H, W) float32 relative inverse depth."""
        from PIL import Image

        H, W = rgb.shape[:2]
        inputs = self.proc(images=Image.fromarray(rgb), return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inputs)
        pred = self.proc.post_process_depth_estimation(out, target_sizes=[(H, W)])[0]["predicted_depth"]
        return pred.float().cpu().numpy()

    def close(self) -> None:
        """Free GPU memory (the VLM stage needs it)."""
        del self.model
        if self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()


def estimate_and_align_depth(out_dir: Path, cfg: dict, fusion_cfg: dict) -> dict[str, dict]:
    """Stage entry: predict, align and save per-frame metric(-ish) depth."""
    import cv2
    import torch

    model_rec = pycolmap.Reconstruction(out_dir / "align" / "sparse")
    img_dir = out_dir / "sfm" / "undistorted" / "images"
    ddir = out_dir / "depth"
    ddir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = cfg["hf_model_ids"][cfg["model_size"]]
    print(f"  Depth model: {model_id} on {device}")
    net = DepthModel(model_id, device)

    rng = np.random.default_rng(cfg["alignment"].get("seed", 0))
    report: dict[str, dict] = {}
    plot_data: list[tuple[str, np.ndarray, np.ndarray]] = []
    for image_id in sorted(model_rec.reg_image_ids(), key=lambda i: model_rec.image(i).name):
        name = model_rec.image(image_id).name
        bgr = cv2.imread(str(img_dir / name), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pred = net.predict(rgb)
        uv, z = sparse_depth_samples(model_rec, image_id)
        rc = pixel_index(uv) if len(uv) else np.zeros((0, 2), np.int64)
        inside = (rc[:, 0] >= 0) & (rc[:, 0] < pred.shape[0]) & (rc[:, 1] >= 0) & (rc[:, 1] < pred.shape[1])
        rc, z = rc[inside], z[inside]
        p = pred[rc[:, 0], rc[:, 1]]
        fit: DepthFit = fit_inverse_depth_affine(p, z, cfg["alignment"], rng)
        report[name] = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in fit.__dict__.items()}
        if fit.ok:
            depth = apply_fit(pred, fit, fusion_cfg["depth_max"])
            np.save(ddir / f"{Path(name).stem}.npy", depth)
            if len(plot_data) < 4 and len(z):
                plot_data.append((name, z, depth[rc[:, 0], rc[:, 1]]))
        else:
            stale = ddir / f"{Path(name).stem}.npy"
            if stale.exists():
                stale.unlink()
    net.close()

    n_ok = sum(r["ok"] for r in report.values())
    for epoch in ("A", "B"):
        tot = [r for n, r in report.items() if n.startswith(f"{epoch}_")]
        ok = [r for r in tot if r["ok"]]
        print(
            f"  [{epoch}] depth aligned for {len(ok)}/{len(tot)} frames; median residual "
            f"{np.median([r['median_rel_residual'] for r in ok]) if ok else float('nan'):.3f}"
        )
    bad = {n: r["reason"] for n, r in report.items() if not r["ok"]}
    for n, why in sorted(bad.items()):
        print(f"    rejected {n}: {why}")
    with open(ddir / "depth_report.json", "w") as f:
        json.dump(report, f, indent=1)
    _plot_alignment(plot_data, ddir / "alignment_plots.png")
    print(f"  {n_ok}/{len(report)} frames usable for fusion")
    return report


def _plot_alignment(data: list[tuple[str, np.ndarray, np.ndarray]], path: Path) -> None:
    """Scatter of aligned monocular depth vs sparse SfM depth (ideal: the diagonal)."""
    if not data:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(data), figsize=(4 * len(data), 4), squeeze=False)
    for ax, (name, z_sparse, z_aligned) in zip(axes[0], data):
        ok = z_aligned > 0
        ax.scatter(z_sparse[ok], z_aligned[ok], s=3, alpha=0.5)
        lim = [0, max(z_sparse.max(), z_aligned.max()) * 1.05]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("sparse SfM depth")
        ax.set_ylabel("aligned monocular depth")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
