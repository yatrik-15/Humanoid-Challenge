"""Evaluation against hand-labelled ground truth.

    python eval/run_eval.py --gt eval/gt/<scene>.json --out out/<scene> [--radius 0.3]

Ground truth for real scenes is made with eval/annotate.py, which triangulates clicked
object centres from the SfM poses only, independent of the change detector:
{
  "scene": "room1",
  "frame": "pipeline",              # coordinates in the run's aligned world frame
  "changes": [
    {"action": "removed", "label": "chair", "centroid": [x, y, z]},
    {"action": "added",   "label": "box",   "centroid": [x, y, z]},
    {"action": "moved",   "label": "bin",   "centroid_before": [...], "centroid_after": [...]}
  ]
}
Synthetic scenes (tests/make_synthetic_videos.py) use "frame": "gt_world" with the true
camera poses; GT is mapped into the run's frame by a similarity transform fitted on true vs.
estimated camera centres (Umeyama).

Metrics for the visibility-aware method (cluster/changes.json) and the naive baseline
(baseline/changes.json):
  change level   a prediction is a true positive if the action matches and its centroid(s)
                 lie within the match radius (moves: both ends) -> precision, recall
  region level   action-agnostic: every box is a region, greedy nearest matching within
                 the radius -> precision, recall
  centroid error mean distance of matched pairs, in the run's world units
Writes eval/results/<scene>.json and .md.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity (s, R, t) minimizing ||dst - (s R src + t)||^2 (Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / xs.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


def gt_to_pipeline(gt: dict, out_dir: Path) -> tuple[list[dict], dict]:
    """Return GT changes in the pipeline frame, plus info about the frame mapping."""
    if gt.get("frame", "pipeline") == "pipeline":
        return gt["changes"], {"frame": "pipeline"}
    if gt["frame"] != "gt_world":
        raise ValueError(f"unknown GT frame {gt['frame']!r}")
    from src.utils.geometry import camera_center

    poses = json.load(open(out_dir / "align" / "poses.json"))
    manifest = {m["frame_name"]: m for m in json.load(open(out_dir / "frames" / "manifest.json"))}
    fps = gt["video_fps"]
    src, dst = [], []
    for name, p in poses.items():
        m = manifest[name]
        key = f"{m['epoch']}_{int(round(m['timestamp_sec'] * fps)):03d}"
        if key in gt["poses"]:
            src.append(camera_center(np.array(gt["poses"][key])))
            dst.append(camera_center(np.array(p["T_cw"])))
    src, dst = np.array(src), np.array(dst)
    s, R, t = umeyama(src, dst)
    resid = np.linalg.norm(dst - (s * src @ R.T + t), axis=1)

    def f(c):
        return (s * R @ np.array(c) + t).tolist()

    out = []
    for c in gt["changes"]:
        if c["action"] == "moved":
            out.append(
                {
                    "action": "moved",
                    "label": c["label"],
                    "centroid_before": f(c["before"]["center"]),
                    "centroid_after": f(c["after"]["center"]),
                }
            )
        else:
            out.append({"action": c["action"], "label": c["label"], "centroid": f(c["center"])})
    info = {
        "frame": "gt_world mapped by Umeyama on camera centres",
        "num_cameras": len(src),
        "scale": float(s),
        "camera_residual_mean": float(resid.mean()),
        "camera_residual_max": float(resid.max()),
    }
    return out, info


def pred_items(changes: list[dict]) -> list[dict]:
    out = []
    for c in changes:
        if c["action"] == "moved":
            out.append(
                {
                    "action": "moved",
                    "centroid_before": c["box_before"]["center"],
                    "centroid_after": c["box_after"]["center"],
                }
            )
        else:
            out.append({"action": c["action"], "centroid": c["box"]["center"]})
    return out


def regions(items: list[dict]) -> list[np.ndarray]:
    r = []
    for c in items:
        if c["action"] == "moved":
            r += [np.array(c["centroid_before"]), np.array(c["centroid_after"])]
        else:
            r.append(np.array(c["centroid"]))
    return r


def change_distance(p: dict, g: dict) -> float | None:
    """Distance between a predicted and GT change if actions agree, else None."""
    if p["action"] != g["action"]:
        return None
    if p["action"] == "moved":
        return max(
            np.linalg.norm(np.subtract(p["centroid_before"], g["centroid_before"])),
            np.linalg.norm(np.subtract(p["centroid_after"], g["centroid_after"])),
        )
    return float(np.linalg.norm(np.subtract(p["centroid"], g["centroid"])))


def greedy_match(D: np.ndarray, radius: float) -> list[tuple[int, int, float]]:
    """Greedy one-to-one matching on a distance matrix (inf = incompatible)."""
    pairs, used_p, used_g = [], set(), set()
    flat = [(D[i, j], i, j) for i in range(D.shape[0]) for j in range(D.shape[1]) if D[i, j] <= radius]
    for d, i, j in sorted(flat):
        if i not in used_p and j not in used_g:
            pairs.append((i, j, float(d)))
            used_p.add(i)
            used_g.add(j)
    return pairs


def score(pred: list[dict], gt: list[dict], radius: float) -> dict[str, object]:
    D = np.full((len(pred), len(gt)), np.inf)
    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            d = change_distance(p, g)
            if d is not None:
                D[i, j] = d
    m = greedy_match(D, radius)
    pr, gr = regions(pred), regions(gt)
    R = (
        np.array([[np.linalg.norm(a - b) for b in gr] for a in pr])
        if pr and gr
        else np.zeros((len(pr), len(gr)))
    )
    mr = greedy_match(R, radius)

    def safe(a, b):
        return round(a / b, 3) if b else None

    return {
        "num_pred": len(pred),
        "num_gt": len(gt),
        "change_TP": len(m),
        "change_precision": safe(len(m), len(pred)),
        "change_recall": safe(len(m), len(gt)),
        "change_centroid_err": round(float(np.mean([d for *_, d in m])), 3) if m else None,
        "region_pred": len(pr),
        "region_gt": len(gr),
        "region_TP": len(mr),
        "region_precision": safe(len(mr), len(pr)),
        "region_recall": safe(len(mr), len(gr)),
        "region_centroid_err": round(float(np.mean([d for *_, d in mr])), 3) if mr else None,
        "matches": [{"pred": i, "gt": j, "dist": round(d, 3)} for i, j, d in m],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True, help="pipeline output dir of the scene")
    ap.add_argument("--radius", type=float, default=None, help="match radius (default: GT file or 0.3)")
    args = ap.parse_args()
    from tabulate import tabulate

    gt = json.load(open(args.gt))
    out_dir = Path(args.out)
    radius = args.radius or gt.get("match_radius", 0.3)
    gt_changes, frame_info = gt_to_pipeline(gt, out_dir)

    methods = {
        "visibility-aware (ours)": out_dir / "cluster" / "changes.json",
        "naive NN baseline": out_dir / "baseline" / "changes.json",
    }
    results = {
        "scene": gt.get("scene", Path(args.gt).stem),
        "out_dir": str(out_dir),
        "match_radius": radius,
        "gt_frame": frame_info,
        "methods": {},
    }
    for name, path in methods.items():
        if not path.exists():
            print(f"  (skipping {name}: {path} missing)")
            continue
        results["methods"][name] = score(pred_items(json.load(open(path))), gt_changes, radius)

    rows = [
        [
            n,
            r["num_pred"],
            r["num_gt"],
            r["change_TP"],
            r["change_precision"],
            r["change_recall"],
            r["change_centroid_err"],
            r["region_precision"],
            r["region_recall"],
            r["region_centroid_err"],
        ]
        for n, r in results["methods"].items()
    ]
    hdr = ["method", "#pred", "#GT", "TP", "P", "R", "centroid err", "region P", "region R", "region err"]
    table = tabulate(rows, hdr, tablefmt="github")
    print(f"Scene {results['scene']}  (match radius {radius}, GT frame: {frame_info['frame']})\n{table}")

    rdir = REPO / "eval" / "results"
    rdir.mkdir(parents=True, exist_ok=True)
    with open(rdir / f"{results['scene']}.json", "w") as f:
        json.dump(results, f, indent=2)
    (rdir / f"{results['scene']}.md").write_text(
        f"# Evaluation: {results['scene']}\n\nmatch radius {radius}; GT frame: {json.dumps(frame_info)}\n\n{table}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
