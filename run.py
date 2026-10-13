"""Detect what changed between two phone videos of the same room.

    python run.py --video_a A.mp4 --video_b B.mp4 --out out/<name>
    python run.py --out out/<name> --stages diff,cluster,semantics,viz

Stages run in order: frames, sfm, align, depth, fusion, baseline, diff, cluster, nav,
semantics, viz. Each stage only reads files written by earlier stages in --out, so any tail of the
pipeline can be re-run on its own.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import yaml

STAGES = [
    "frames",
    "sfm",
    "align",
    "depth",
    "fusion",
    "baseline",
    "diff",
    "cluster",
    "nav",
    "semantics",
    "viz",
]
REPO = Path(__file__).resolve().parent


def load_config(config_path: str) -> dict:
    """Load pipeline configuration from YAML file (UTF-8)."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def banner(stage: str, title: str) -> None:
    print(f"\n[{stage}] {title}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="3-D change detection between two videos of a room")
    parser.add_argument("--video_a", type=str, help="'before' video (needed for the frames stage)")
    parser.add_argument("--video_b", type=str, help="'after' video (needed for the frames stage)")
    parser.add_argument("--out", type=str, required=True, help="Output directory, e.g. out/scene1")
    parser.add_argument("--config", type=str, default=str(REPO / "config.yaml"))
    parser.add_argument(
        "--stages", type=str, default="all", help=f"'all' or a comma-separated subset of: {','.join(STAGES)}"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    # Keep model downloads off the (small) system drive; must be set before HF imports.
    hf_home = config.get("paths", {}).get("hf_home")
    if hf_home:
        os.environ.setdefault("HF_HOME", str((REPO / hf_home).resolve()))

    stages = STAGES if args.stages == "all" else [s.strip() for s in args.stages.split(",")]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        sys.exit(f"error: unknown stage(s) {unknown}; valid: {STAGES}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"config {args.config}\noutput {out_dir}\nstages {','.join(stages)}")
    t_start = time.time()

    if "frames" in stages:
        banner("frames", "extracting frames")
        if not (args.video_a and args.video_b):
            sys.exit("error: --video_a and --video_b are required for the frames stage")
        for v in (args.video_a, args.video_b):
            if not Path(v).exists():
                sys.exit(f"error: video not found: {v}")
        from src.frames import extract_frames

        extract_frames(Path(args.video_a), Path(args.video_b), out_dir, config["frames"])

    if "sfm" in stages:
        banner("sfm", "joint SfM (DISK + LightGlue, pycolmap)")
        from src.sfm import run_joint_sfm

        run_joint_sfm(out_dir, config["sfm"])

    if "align" in stages:
        banner("align", "gravity alignment and scale")
        from src.alignment import align_scene

        align_scene(out_dir, config["alignment"])

    if "depth" in stages:
        banner("depth", "monocular depth, aligned per frame")
        from src.depth import estimate_and_align_depth

        estimate_and_align_depth(out_dir, config["depth"], config["fusion"])

    if "fusion" in stages:
        banner("fusion", "TSDF fusion of A and B")
        from src.fusion import fuse_depths

        fuse_depths(out_dir, config["fusion"])

    if "baseline" in stages:
        banner("baseline", "naive nearest-neighbour baseline")
        from src.baseline import naive_changes

        naive_changes(
            out_dir,
            config["baseline"],
            config["cluster"],
            config["moved"],
            config["diff"]["evidence_voxel_size"],
        )

    if "diff" in stages:
        banner("diff", "render-and-compare change detection")
        from src.diff import detect_changes

        detect_changes(out_dir, config["diff"], config["fusion"])

    if "cluster" in stages:
        banner("cluster", "clustering and move pairing")
        from src.cluster import cluster_changes

        cluster_changes(out_dir, config["cluster"], config["moved"], config["diff"]["evidence_voxel_size"])

    if "nav" in stages:
        banner("nav", "route check on the changed room")
        from src.navigation import plan_routes

        plan_routes(out_dir, config["navigation"], config["viz"]["obstacle_z_band"])

    if "semantics" in stages:
        banner("semantics", "VLM labels")
        from src.semantics import label_changes

        label_changes(out_dir, config["semantics"], config["fusion"])

    if "viz" in stages:
        banner("viz", "reports and visualizations")
        from src.viz import generate_visualizations

        generate_visualizations(out_dir, config["viz"])

    print(f"\ndone in {time.time() - t_start:.0f} s -> {out_dir}")


if __name__ == "__main__":
    main()
