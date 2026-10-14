# DEVLOG — What Changed in the Room?

## Stage 0: Environment Report + Repo Skeleton

**Date:** 2026-10-10

### Environment Report
| Property       | Value                                     |
|----------------|-------------------------------------------|
| OS             | Windows 11 Home, Build 26200              |
| Python         | 3.13.2                                    |
| GPU            | NVIDIA GeForce RTX 4050 Laptop, 6 GB VRAM |
| CUDA UMD       | 13.3, Driver 610.88                       |
| RAM            | ~24 GB                                    |
| Disk (Y:)      | ~63 GB free                               |
| ffmpeg         | NOT installed (will use imageio[ffmpeg])   |
| Internet       | OK                                        |

### Compatibility Checks
- `pycolmap 4.2.1`: cp313-win_amd64 wheel available ✓
- `open3d 0.20.0`: cp313-win_amd64 wheel available ✓

### Decisions
- **Python 3.13** is supported by both pycolmap and open3d. No downgrade needed.
- **Project moved to Y:\Humanoid** due to limited disk space on C: (~50 GB vs ~63 GB on Y:).
- **ffmpeg** not on PATH; will install `imageio[ffmpeg]` which bundles its own ffmpeg binary.
- **Depth model**: will use `small` (vits) variant of Depth Anything V2 to fit in 6 GB VRAM.
- **VoxelBlockGrid** (Open3D tensor API) will be used for TSDF fusion — need to verify exact
  API signatures from open3d 0.20.0 at Stage 4.

### Repo Structure Created
`
Y:\Humanoid\
├── run.py
├── config.yaml
├── src/
│   ├── __init__.py
│   ├── frames.py
│   ├── sfm.py
│   ├── alignment.py
│   ├── depth.py
│   ├── fusion.py
│   ├── diff.py
│   ├── cluster.py
│   ├── semantics.py
│   ├── viz.py
│   └── utils/
│       ├── __init__.py
│       └── geometry.py
├── tests/
│   └── __init__.py
├── eval/
│   └── gt/
├── data/
├── examples/
└── out/
`

### Status: PASS
- Environment documented with verified values (not assumed).
- All directories and stub files created.
- config.yaml and run.py in place.

---

## Stage 1: Frame Extraction

**Date:** 2026-10-10

### Commands Run
1. `pip install imageio[ffmpeg] imageio[pyav] opencv-python-headless numpy pyyaml`
   - imageio 2.38.1, opencv-python-headless 5.0.0.93, numpy 2.5.3, pyyaml 6.0.3, av 19.0.1
2. Generated synthetic test videos: `test_A.mp4`, `test_B.mp4` (30 frames each, 10fps, 480x640)
3. Ran frame extraction with `target_fps=2.0`, `blur_threshold=50.0`

### Evidence
```
  [A] Read 6 candidate frames, kept 6, skipped 0 blurry
  [B] Read 6 candidate frames, kept 6, skipped 0 blurry
  Total frames: 12 (A: 6, B: 6)
  manifest.csv exists: True
  Frame files exist: True
  STAGE 1 TEST: PASS
```

### Issues
- `imageio[ffmpeg]` plugin uses `imageio-ffmpeg` (subprocess-based ffmpeg), but for reading/writing
  video via Python API, `pyav` plugin is needed. Added `imageio[pyav]` to requirements.
- `iio.imwrite` needs explicit `codec='libx264'` with pyav plugin for mp4.
- Blur filter did not trigger on synthetic videos (expected — they are perfectly sharp).
  Will verify on real phone videos later.

### Decisions
- Using `imageio` with `pyav` plugin instead of `ffmpeg` subprocess for better API control.
- Frame naming: `{epoch}_{counter:04d}.jpg` (1-indexed counter per epoch).

### Status: PASS

---

## Dependencies installed (2026-10-13)

`pip install open3d pydantic transformers huggingface-hub rerun-sdk scikit-learn matplotlib tabulate Pillow pytest`
(the first attempt failed with "No matching distribution found for pydantic"; `pip index versions`
then listed 2.14.1, so it was a transient index/network failure, not a Python 3.13 issue).

Installed: open3d 0.20.0, pydantic 2.14.1, transformers 5.19.0, huggingface_hub 2.2.0,
rerun-sdk 0.39.0, scikit-learn 1.9.1, matplotlib 3.11.2, tabulate 0.10.0, pytest 9.1.1.
Already present: torch 2.6.0+cu124 (CUDA available), pycolmap 4.2.1, hloc 1.5, lightglue.
**open3d 0.20.0 wheel is CPU-only on Windows** (`open3d.core.cuda.is_available() == False`).

---

## Stage 2: Joint SfM — code reviewed, NOT RUN (waiting for real videos)

The synthetic `test_A/B.mp4` are 2-D drawings with no camera motion; SfM cannot be validated on them.

### API verification (installed versions)
- `hloc.reconstruction.main(sfm_dir, image_dir, pairs, features, matches, camera_mode, verbose,
  skip_geometric_verification, min_match_score, image_list, image_options, mapper_options)`.
  This hloc moves `frames.bin`/`rigs.bin`, i.e. it targets pycolmap 4.
- pycolmap 4.2.1: `Reconstruction.reg_image_ids()` and `Image.cam_from_world()` are METHODS;
  `Rigid3d.matrix()` is 3x4 world->camera; `IncrementalPipelineOptions` has `min_num_matches`
  and `ba_global_max_num_iterations`.
- hloc adds +0.5 to keypoints before COLMAP (`hloc/triangulation.py:53  keypoints += 0.5  # COLMAP origin`):
  COLMAP image coordinates put pixel (r, c)'s centre at (c+0.5, r+0.5).

### Fixes to src/sfm.py
1. `image_list` was not passed to `reconstruction.main` -> pycolmap would import `manifest.csv/json`
   from the frames folder. Now passed.
2. `min_match_score` and all `mapper:` keys from config were ignored. Now passed through.
3. `reconstruction.main` returns None on failure -> now raises a clear RuntimeError.
4. Added: `pycolmap.undistort_images` -> PINHOLE model (`sfm/undistorted/`), `poses.json` export
   (T_cw, K, size per image), `trajectories.png` (A and B camera centres in one frame).

---

## Stage 3 (code + unit tests; real-data test pending)

`src/alignment.py`: RANSAC floor fit + `pycolmap.Sim3d` applied with `Reconstruction.transform`
(verified: Sim3d(scale, Rotation3d, t) maps X -> s R X + t). Sanity check built in: mean
reprojection error must be unchanged by the similarity transform, else RuntimeError.
- Floor candidate must (a) be within `max_tilt_deg` of the mean camera-up, (b) have the cameras
  above it, (c) have <= `max_below_ratio` of points below it. (c) rejects table tops.
- Thresholds are relative to scene scale L (median point distance to the camera centroid).
- **Scale decision:** default `scale_mode: camera_height` = APPROXIMATE metres from an
  ASSUMED hand-held camera height (`assumed_camera_height_m: 1.3`). Recorded in
  `align/alignment.json` as `scale_is_metric` + `scale_note`. Falls back to `normalized`
  (not metric) if no floor is found.

`src/depth_align.py`: per-frame 2-point RANSAC fit of inv_z = a*pred + b against the frame's
sparse points, LSQ refine, reject on few points / low inlier ratio / high median residual.
(The Depth Anything model loader itself is not written yet — model ID still UNVERIFIED.)

---

## Stage 5: Synthetic render-and-compare test — PASS

**Command:** `venv\Scripts\python.exe -m pytest tests -q`
**Result:** `32 passed in 10.25s`

### Bugs found by the tests (all fixed)
1. `look_at_T_cw` built cameras upside-down (`x = cross(z, -up)` gives camera y = world up;
   OpenCV needs y down). Caught by `test_look_at_world_up_is_image_up`; also broke 2 alignment
   tests. Fix: `x = cross(z, up)`.
2. My assumption "Open3D puts pixel centres at integers" was WRONG (see probes below).

### Open3D 0.20.0 probes (scratch scripts, exact synthetic depth, 2 cm voxels)
- `integrate(...)` with `extrinsic = T_cw` works; stored tsdf is normalized to [-1, 1]
  (observed range -0.99999905..1.0).
- `voxel_coordinates_and_flattened_indices()`: coords / voxel_size are integral -> voxel ijk
  samples world position ijk * voxel_size. `attribute()` is sized by block CAPACITY (81.9 M
  entries for 20000 blocks); index it with the returned flat indices.
- Pixel convention, regressing render error on image-space depth gradient:

  | K given to integrate | median abs err | bias | fitted shift u / v |
  |---|---|---|---|
  | K (unchanged) | 2.9 mm | -0.2 mm | +0.004 / -0.013 px |
  | K - 0.5 | 19.5 mm | +4.8 mm | -0.242 / -0.591 px |
  | K + 0.5 | 16.4 mm | -3.7 mm | +0.234 / +0.524 px |

  -> Open3D integrate uses the COLMAP convention; K is passed unchanged.

### FAILED planned approach: `VoxelBlockGrid.ray_cast`
Same grid, same pose, isolated per process with timeouts:

| range_map_down_factor | result |
|---|---|
| 1 | hung (> 60 s) |
| 2 | ran; hit sets differ between repeated calls; 68–165 px per view with >10 cm error |
| 4 | finished in one run (0.03 s), hung (> 60 s) in another |
| 8 (default) | first call incomplete: 8060 hits vs 13003 after 5 calls ("Could not generate full range map" warning) |
| 16 | hung (> 40 s) |

An incomplete range map can skip near blocks and return a FARTHER surface -> fake change.
**Replacement (renderer only; architecture unchanged — still render grid A vs grid B from the
same poses):** extract each grid's zero-level mesh (`extract_triangle_mesh`) and cast our own
rays (`utils.geometry.pixel_rays`, camera-z-normalized so t_hit = z-depth) with
`t.geometry.RaycastingScene`. Measured: identical output over 4 repeated calls, ~1 ms per
160x120 cast, 2.9 mm median error vs exact depth. The mesh covers every allocated block, so
removed objects are still rendered.

### Design additions found necessary by the synthetic test
1. **Free-space confirmation.** Render-vs-render alone flags a false REMOVED wherever the other
   epoch's ray passes through space it never observed. A candidate is confirmed only if the
   OTHER grid has weight > 0 and tsdf > `free_tsdf_min` at the evidence point; otherwise UNKNOWN.
2. **Free-space carving.** Open3D allocates blocks only near surfaces, so empty space far from
   any surface stays "unobserved" and the top of a removed tall object could never be
   confirmed. `carve_free_space` also allocates blocks along a subsample of each frame's rays;
   integrate() then writes tsdf = +1 there. Tested by box6 (1.2 m pole, REMOVED, detected
   with z-extent > 0.8).
3. **Depth-edge erosion.** Rays just beside a silhouette carve the object's edge as free
   (projective TSDF). Measured on the unobserved-box4 scenario: without erosion 639 of 3042
   REMOVED candidates passed the free-space check (false REMOVED); with 1 px erosion, 1 of 2083.
4. **View groups.** `min_views` counts distinct viewpoint groups (camera centres within
   `view_group_radius_rel` * scene scale and optical axes within `view_group_angle_deg`).

### What the synthetic test asserts (tests/test_change_detection.py)
Scene A has box1 (removed), box2 (static), box4 (static but B never observes its volume while
seeing the floor behind it), box5 (moved), box6 (tall, removed). B adds box3. Different
camera orbits for A and B.
- box1, box6 REMOVED; box3 ADDED; box5 paired as MOVED (appearance + size).
- box2 not reported (pure viewpoint change); box4 not reported and present in UNKNOWN.
- Exactly 3 removed and 2 added clusters.
- **Ablation inside the test:** with `free_space_check: false`, box4 IS reported REMOVED,
  so the box4 assertion passes only because of the free-space check.

Ablation printout (before carving/erosion were added; scratch `ablate_synth.py`):
full method = exactly the planted changes; no free-space check = +4 spurious REMOVED
clusters (thin synthetic walls seen from both sides); `min_views=1` = +2 spurious;
naive nearest-neighbour baseline (5 cm) = 9 spurious clusters and the moved box reported
as both removed and added on each side.

### Known limitations (so far)
- Ray-mesh rendering misses ~7% of exact-hit pixels (silhouettes, low-weight areas);
  these are UNKNOWN, never changes.
- Synthetic images are 160x120; `edge_erode_px` for real images (default 2) is UNVERIFIED.

### Status: PASS

---

## Synthetic END-TO-END run (all stages, before real videos exist)

`tests/make_synthetic_videos.py` renders a closed textured room (3-D value-noise solid
texture so DISK has features) as two 640x480 "videos" (60 frames each, 4 fps) along
different hand-held-like arcs, with known changes: chair REMOVED, box ADDED, bin MOVED,
plus static table/sofa/cabinet. Exact GT poses + boxes in `data/synth_gt.json`.
This tests the code path end to end; it is NOT the real-data acceptance test.

```
python -m tests.make_synthetic_videos --out data/synth
python run.py --video_a data/synth_A.mp4 --video_b data/synth_B.mp4 --out out/synth
python eval/run_eval.py --gt data/synth_gt.json --out out/synth
```

### Bugs found and fixed on the way
1. hloc `superpoint` extractor imports `SuperGluePretrainedNetwork` (git submodule, not
   installed by pip). -> default `feature_type: disk` (hloc conf `disk`, kornia 0.8.3) with
   `disk+lightglue`. Spec allows DISK or SuperPoint.
2. `match_features.main(features=<Path>)` requires an explicit `matches` path (read hloc
   source: raises "Either provide both features and matches as Path or both as names").
   This bug was in the original sfm.py; fixed.
3. LightGlue weight download (github releases) failed twice with WinError 10053/10054 while a
   4.4 GB HF download ran; fetched `disk_lightglue_v0-1_arxiv.pth` (47.6 MB) with a retry
   loop into the torch hub cache. Transient network issue, no code change.

### Results (from out/synth/*)
- SfM: 60/60 registered (A 30/30, B 30/30), 18735 points, mean reprojection error
  **0.767 px**, mean track length 12.37; focal 520.11 vs true 520.
- Pose accuracy (eval Umeyama on camera centres, GT -> pipeline frame): mean residual
  **1.4 mm**, max 2.8 mm; fitted scale 1.017 (the camera-height scale assumption gave ~1.7%
  scale error here).
- Alignment: floor found, 55.5% inliers, 6.0 deg from camera-up.
- Depth: 60/60 frames passed alignment, median relative residual 0.010 per epoch.
- Depth Anything V2 Small (verified): DPTImageProcessor + DepthAnythingForDepthEstimation,
  output corr +0.909 with 1/z and -0.931 with z (relative INVERSE depth), 244 MB VRAM peak.

### False positives found and the fix
First run: ours 9 predictions (3 TP), naive baseline 32. Diagnosis (scratch `diag_fp.py`,
mapping each cluster back to the synthetic world): every FP lay on a STATIC surface (walls,
table, sofa), was a thin sheet (smallest extent 0.03-0.06) and/or hugged the other
epoch's surface (median gap 0.018-0.046 for the strips). TPs: smallest extent 0.31-0.46,
gap 0.138-0.530. Mechanism: monocular depth places a static surface slightly differently
from A's vs B's viewpoints. Gap alone does NOT separate (wall sheets 0.089-0.139 vs TP 0.138).
Added two artefact filters in `cluster_points` (applied identically to the baseline):
  - sheet: smallest principal extent < 0.10 AND middle extent > 3 x smallest (poles kept —
    the first version "thickness only" rejected the Stage 5 1.2 m pole; caught by the test).
  - hugging: median distance to the other epoch's surface < 0.06.
These were DESIGNED FROM THIS ONE SYNTHETIC SCENE; the negative control (Stage 7) on real
data is their real test.

### Evaluation (eval/results/synth.md, match radius 0.3)
| method | #pred | #GT | TP | P | R | centroid err | region P | region R |
|---|---|---|---|---|---|---|---|---|
| visibility-aware (ours) | 3 | 3 | 3 | 1.0 | 1.0 | 0.083 | 1.0 | 1.0 |
| naive NN baseline | 15 | 3 | 3 | 0.2 | 1.0 | 0.085 | 0.25 | 1.0 |

### Semantics (Qwen2-VL-2B, fp16 on CUDA — fitted in 6 GB)
First run labeled all three changes "green block": view selection used only 9 box corners
for occlusion and preferred huge close-ups (removed chair cut off; added box crop centred on
the bin), and the prompt did not say which image contains the object, and contained
coordinates that the model copied into its summary.
Fixes: occlusion test on the object's fused surface points; penalty for boxes cut by the
image border; area capped at 35% of the image; yellow outline drawn on both crops; prompt
states which image shows the object and contains NO numbers; summaries with any digit
are discarded. Second run: moved -> "grass block", removed -> "red cube", added -> "cube"
(crops checked visually: correct objects, correct before/after).

### Visual checks
- floor_map.png: room outline, table, sofa, both camera paths, change footprints.
- side_by_side.mp4 (30 frames): projected 3-D boxes land on the objects in both A and B
  frames -> conventions consistent from SfM through alignment, clustering and reprojection.

**Tests: `pytest tests` -> 38 passed.**

---

## Real data: `data/Video_A.mp4` / `data/Video_B.mp4` (2026-10-13)

Phone: Nothing A063 (Android 15), 1920x1080 stored, ~59-60 fps, 30.6 s (A) / 32.2 s (B).
Scene: bed + headboard shelf with toy cars/figures, posters, bedside table. Visual check of
frames: A has NO black bottle; B has a black bottle on the left pillow. So A->B = 1 ADDED.
(Ground truth for eval is annotated by hand, separately from the detector.)

### Stage 1 on real data — PASS (after one bug fix)
- First run: frames came out rotated 90 deg. PyAV `VideoFrame.rotation` = -90 (display
  matrix), which imageio's `imiter` ignores. Checked visually on B_0041: rotating 90 deg
  CLOCKWISE gives the upright image. `src/frames.py` now reads the rotation with PyAV and
  applies it (270 -> CW verified; 90/180 by the same convention, unverified on real data).
  Also deletes stale A_/B_ jpgs before re-extracting.
- Re-run: `run.py --video_a data/Video_A.mp4 --video_b data/Video_B.mp4 --out out/bottle --stages frames`
  -> A 61 kept / 61, B 65 kept / 65, 900x1600 portrait frames.
- Blur filter dropped nothing: Laplacian variance (on the 1600-px frames) min/median/max
  A 171.6 / 407 / 494.9, B 151 / 485.2 / 606.6, all above the 50 threshold. Videos are
  steady; the threshold is not tuned (no blurry frames exist to tune it on).

### Stage 2 on real data — PASS
`run.py --out out/bottle --stages sfm` (DISK 4096 + LightGlue, exhaustive 7875 pairs, 1068 s total;
log `out/logs/bottle_sfm.log`). From `out/bottle/sfm/sfm_report.json`:
- Registered 126/126 (A 61/61, B 65/65) in ONE model; 16277 points; mean reprojection error
  **0.978 px**; mean track length 31.09; one shared SIMPLE_RADIAL camera, f = 1468.2 px, k = 0.034.
- Cross-epoch link (pycolmap check): 15221 / 16277 points (93.5%) have observations in BOTH
  A and B frames -> A and B share one frame through real matches, not a coincidental merge.
- `trajectories.png`: two roughly parallel lateral sweeps along the bed, B offset from A.

### Stage 3 on real data — PASS after a scale-mode change
- First run (`scale_mode: camera_height`): "floor" found, 46.6% inliers, 8.6 deg from camera-up.
  Overlaying inliers (|z| < 5 cm) on frames showed the plane is the **mattress top** (plus the
  bedside table); the room floor is never visible. Gravity is still right (mattress is
  horizontal), but the assumed 1.3 m camera height was measured from the bed -> visible
  mattress area came out 2.87 x 1.82 m, implausible.
- Probe (12 frames, Depth-Anything-V2-Metric-Indoor-Small-hf; repo verified with HfApi,
  config `depth_estimation_type: metric`, `max_depth: 20`): prediction correlates 0.89-0.98
  with SfM z (so it is z-depth, not inverse); per-frame median metres/unit 0.0164-0.0404,
  median 0.0353. Low outliers = first frames of each video (close-up blank wall / bed).
- Switched to **`scale_mode: metric_depth`** (new, now the default): s = median over every 3rd
  frame of median(pred / SfM z). Real pair: **s = 0.03531 m/unit from 42 frames**
  (min 0.01642, max 0.04157, IQR/median 0.103), vs 0.0456 from camera height (~29% larger).
  Median camera height above the bed plane: 1.007 m. Stats in `align/alignment.json`.
- NOTE: z = 0 is the support plane found (here the mattress top), not necessarily the floor.
  The floor map's `obstacle_z_band` is relative to that plane.
- The synthetic end-to-end result above was produced with `camera_height` (floor visible there).
- Depth (relative V2-Small, per-frame inverse-depth affine fit): 126/126 frames accepted;
  median rel. residual A 0.021, B 0.016; worst A_0044 0.030; inlier ratio min 0.990, median
  0.9995. `depth/alignment_plots.png`: points on the diagonal over 1.5-3.0 m.
- `pytest tests` -> 38 passed after the change.

### Stage 4 on real data — PASS
`--stages fusion`: A 61 frames / 634 blocks / 49113 surface pts; B 65 / 678 / 50842 (24.7 s).
Static-background overlap (Open3D `compute_point_cloud_distance`): A->B median 1.00 cm,
p90 3.1 cm, 95.5% < 5 cm; B->A median 1.05 cm, p90 3.6 cm, 94.1% < 5 cm. Front and top
scatter plots: both walls and the bed plane of A and B coincide.

### Stage 6 on real data (`--stages baseline,diff,cluster`)
- Diff: 126 poses rendered, **5 view groups** (camera path spans only ~0.56 x 0.24 m, so
  most poses fall in the same group); confirmed voxels 85 added, 1 removed, 432 unknown.
- First clustering: **0 changes -> bottle MISSED.** Diagnosis: DBSCAN cluster at
  (-0.22, 1.97, 0.18), 28 voxels, principal extents 0.028 / 0.059 / 0.204 m, gap to A
  0.083 m, 3-4 view groups = the bottle (matches its place in B frames). Killed by
  `min_cluster_volume: 0.001` (box ~3.6e-4 m^3): evidence is the visible SHELL, ~1 voxel
  thick, so box volume under-measures solid objects.
- Fix: size test on the two largest principal extents, `min_frontal_area: 0.005` m^2
  (~5 x 10 cm face). Real pair after fix: **1 ADDED** at (-0.22, 1.98, 0.19), box
  0.068 x 0.042 x 0.209, 4 view groups. Naive baseline: 1 moved, 0 removed, 8 added.
- **Trade-off (not hidden):** on the synthetic scene this lets through one extra REMOVED
  cluster on the back wall (23 voxels, extents 0.009/0.078/0.187, gap 0.065), previously
  removed by the volume test. Synthetic eval now: ours P 0.75 / R 1.0 (was 1.0 / 1.0),
  baseline P 0.176 / R 1.0 (eval/results/synth.md). Its features overlap the real bottle's
  except thickness (0.9 cm planar vs 2.8 cm curved); I did NOT add a planarity rule from
  one sample of each. The real negative control (Stage 7) must decide.
- `pytest tests` 38 passed (tests updated to the new key).

### Stage 8/9 on real data (semantics, viz)
- Views chosen: A_0021 / B_0029. Crops checked visually: B crop = black bottle inside the
  outline, A crop = empty pillow at the same place -> projection consistent.
- First VLM run: `label_unavailable`. Raw replies were valid JSON with `"action": "placed"`,
  identical 3 times (greedy decoding + identical retry prompt). Schema kept strict; retries
  now append the pydantic validation error (`RETRY_SUFFIX`). New test
  `test_retry_prompt_carries_the_validation_error`; `pytest tests` -> 39 passed.
- Re-run: attempt 1 "placed", attempt 2 valid -> **"cup", conf 0.8**, action agreed.
  Honest assessment: the object is a black insulated bottle/tumbler; "cup" is close but
  imprecise, and the summary says "couch" (it is a bed). Qwen2-VL-2B limitation.
- Report wording fixed: z=0 is "the dominant horizontal support plane" (here the mattress),
  not necessarily the floor.
- side_by_side.mp4 (65 frames): green box drawn on the bottle in B_0033; matched A frame
  shows the empty pillow. floor_map.png: change on the bed by the headboard wall.

### Still open
- Stage 7 negative control: needs two new recordings of an unchanged scene.
- Stage 10: needs hand-labelled ground truth `eval/gt/bottle.json`.

## Second batch of real recordings (2026-10-13)

Recorded: `Change_A/B` (keyboard removed from the bed), `New_change_A/B` (green ball
added on a tiled floor, different room), `Static_A/B` (nothing changed, different path).
Visual check of frames: Change A has the keyboard on the bed, B does not; the black F1 car
lies on the bed next to the pillow in BOTH (no visible move). Static A/B: shelf, pillows and
bed look the same.

### Stage 1 bug: absolute blur threshold
`New_change`: the fixed threshold (variance of Laplacian 50) dropped 46/60 and 49/51 frames.
They are sharp: a plain glossy tiled floor scores 40-60, the bed scenes 300-500. Replaced by
`blur_keep_mask`: keep a frame if score >= `blur_rel_threshold` (0.5) x the median of
`blur_window` (7) neighbouring candidates (texture changes slowly, motion blur is a dip).
Re-extraction: change 75/75 + 64/64, new_change 58/60 + 51/51, static 75/75 + 74/74.
Tests: `tests/test_frames.py`.

### Full runs (`--stages sfm,...,viz`; logs `out/logs/<pair>_run.log`)
| pair | registered (A / B) | reproj px | scale m/unit | changes (ours) | naive baseline |
|---|---|---|---|---|---|
| static | 149/149 (75/75, 74/74) | 1.076 | 0.0249 | **0** (3 added + 0 removed voxels confirmed, 47 unknown) | 4 added |
| change | 116/139 (63/75, 53/64) | 1.042 | 0.0264 | **0** (keyboard missed) | 1 added (spurious) |
| new_change | 65/109 (37/58, 28/51) | 1.843 | 0.0189 | invalid (Stage 2 FAIL) | invalid |

### Stage 7 negative control — PASS
Static pair: 0 confirmed clusters (visibility-aware) vs 4 spurious clusters (naive).

### Keyboard (change pair) — missed; documented limitation
- The 20 confirmed REMOVED voxels are a vertical strip at x=-0.83, z 0..0.42, 1.2 cm from
  B's surface (static-edge artefact), not the keyboard; DBSCAN finds no cluster.
- Cause: a flat keyboard is ~2-3 cm tall; tau = max(0.05 x depth, 0.03) is ~7.5 cm at the
  ~1.5 m filming distance.
- tau sweep on all valid pairs (scratch script, diff+cluster re-run on copies of align/fusion):

  | tau_rel : tau_abs_min | bottle (1 real) | static (0 real) | change (1 real) |
  |---|---|---|---|
  | 0.05 : 0.03 (default) | 1, correct | 0 | 0 |
  | 0.03 : 0.02 | 2 (bottle + 1 false) | 0 | 0 |
  | 0.02 : 0.015 | 0 | 1 false | 0 |
  | 0.015 : 0.01 | 0 | 1 false | 0 |

- At 0.02 : 0.015 no evidence cluster exists at bed height (z ~ 0-0.05); the clusters found
  are wall/headboard sheets hugging B's surface (rejected). Conclusion: objects lower than
  ~5 cm lying on a surface are below the geometric resolution of monocular depth + 2 cm TSDF
  here. Default kept (best on all three pairs). Not fixed by tuning.

### Stage 2 FAIL on new_change (ball) — not processed further
65/109 registered, 1.84 px, no support plane found, depth alignment accepted only 13/65
frames. Frames show mostly a glossy, nearly textureless tiled floor with mirror reflections
(reflections move with the viewpoint -> inconsistent matches) and a TV whose screen content
differs between A and B. Downstream results for this pair are not meaningful.

## Cleanup before the first commit (2026-10-13)

- Lint/format: added `pyproject.toml` (ruff, line length 110); `ruff check --fix`, `ruff format`.
  E501 is left to the formatter (long string literals such as the VLM prompt stay intact).
  `fusion.py`: the per-epoch `frames()` generator now takes `names` explicitly (ruff B023;
  it was consumed inside the loop, so not a live bug).
- Comments and docstrings rewritten to say what the code does and why; measurement history
  stays in this log. Config values unchanged (parsed YAML compared equal before/after).
- `src/frames.py` rewritten more compactly; re-extracting the keyboard pair gives an
  identical manifest and identical pixels (sampled frames, max abs diff 0).
- New test `test_integrate_uses_colmap_pixel_centres` replaces the earlier scratch probe:
  median render error 2.23 mm with K unchanged vs 18.26 / 18.09 mm with K shifted by -0.5 / +0.5 px.
- New `eval/annotate.py` (ground truth by clicking an object in 2+ frames, DLT-triangulated
  from SfM poses only) with a round-trip test. `pytest` -> 43 passed.
- Removed the early 2-D toy videos (`tests/gen_test_videos.py`, `data/test_*.mp4`,
  `out/test_synth`); run logs moved to `out/logs/`. Stripped UTF-8 BOMs from 8 files.
- Re-ran baseline..viz on the bottle pair after the refactor: same result (1 added, "cup"
  0.8; baseline 9).
- README rewritten around the real results; `examples/` holds the reports of the bottle,
  static and keyboard pairs plus one side-by-side frame and the bottle floor map.

## Route check and look-next idea (2026-10-13)

- Looked at a "where to look next" suggestion built from the unknown evidence. On the bottle
  pair the largest unknown region (312 of 432 voxels) is blank white wall above the bed
  (checked by projecting it into B_0053 / B_0033); static and keyboard pairs have no
  unknown cluster at all. A second look would not resolve textureless wall for monocular
  depth, so the idea was dropped.
- New `nav` stage (`src/navigation.py`): replays A's camera path on B's occupancy grid
  (points in `viz.obstacle_z_band`, cells 5 cm, inflated by a 0.25 m robot radius), blames
  newly blocked route cells on added/moved-after boxes, and replans with A*. Cells that
  were already blocked in A are not counted (handheld paths pass close to furniture: 136
  such cells on the synthetic room).
- Skipped unless the floor was found and the median camera height is 1.1-1.9 m. All three
  bedroom pairs are skipped (camera 0.94-1.01 above z = 0, which is the mattress or a
  fallback plane), as they should be. Synthetic room: route 9.15 m, 0 cells newly blocked,
  which matches the layout (no change sits on A's path). Tests: `tests/test_navigation.py`
  (blocked corridor is blamed on the added box and replanned with >= 0.25 m clearance;
  unchanged room keeps the route). `pytest` -> 47 passed.
- Found stale labels in `out/synth`: `semantics/labels.json` predated the
  `min_frontal_area` change, so ids pointed at different changes (#2 "removed" carried the
  added cube's text). Re-ran semantics; `viz` now drops any label whose action differs
  from its change.

## Walkway pair: bag moved, ball moved, bottle added (2026-10-13)

`data/Multi_{A,B}.mp4`, about 35 s each, phone pitched about 35 deg down to keep the floor in
view. Glossy white tiles, black window at night. Logs `out/logs/multi_run{,2,3}.log`.

- SfM: 138/138 registered (73 A, 65 B), 1.41 px, 35559 points.
- First run: floor not found. Cause: `estimate_camera_up` used the mean image-up vector,
  which is off by the pitch angle when the phone looks down; the floor normal was 34.5 deg
  from it (limit 30). Fix: up is refined as the direction most orthogonal to all camera
  x-axes (handheld phones hardly roll). Change vs image-up on the pairs: walkway 29.8 deg,
  bottle 6.5, static 10.2, keyboard 9.7, synthetic 6.0. Test with a 35 deg pitched camera.
- The floor plane then held 3.9 % of sparse points (glossy tiles give few points among a lot
  of furniture; about 1400 points absolute), camera 1.30 m above it, point heights peaking
  at 0-0.1 m (floor) and 0.6 m (bed). `floor_min_inliers_ratio` 0.05 -> 0.03; static and
  keyboard pairs have no candidate plane at all, bottle has 46.6 %.
- Second run: bag moved 1.08 m (correct), 6 more changes. The largest (1.6 x 1.0 m "added")
  was the far floor: fused floor height along the walkway at y 1.4-2.6 m was -0.08..+0.09 in
  A and +0.11..+0.14 in B (monocular depth on reflective tiles). Fix: `floor_snap` puts
  pixels within [-0.4, 0.2] m of z = 0 with normals within 30 deg of vertical on the plane,
  only when `floor_is_reliable` (floor found, cameras 1.1-1.9 m above it). Checked visually:
  the mask covers the floor and none of bag, ball, vase, bottle or bed. 36-39 % of pixels
  snapped. Test: a floor rendered 8 % too deep is restored, an upright face is untouched.
- Third run: 1 moved (bag, 1.10 m), 1 removed + 3 added at the window end (vase, window
  frame), ball not detected, bottle not reported (A never saw that spot; no evidence of any
  kind there). Route check: A's route (3.11 m) blocked by the bag (change 0); no way
  around within the observed area.
- Why the window end fails: rendered A and B from the same poses disagree beyond tau on
  10-20 % of pixels at 1.5-3 m even with the bag's footprints excluded, against 99th
  percentile relative difference 1.6 % on the static pair. B never got closer than about
  2.5 m to the window (the bag was in the way). The ball (about 15 cm) is the size of this
  error, and the floor prior also flattens part of it. Not tuned further.
- A grazing-angle depth filter was tried and dropped: it flags real side-on surfaces
  (wardrobe and bed sides), not the floor error.
- Re-ran align..viz on bottle, static and keyboard with the new up estimate: bottle 1 added
  ("cup"), baseline 9; static 0, baseline 4; keyboard 0, baseline 1. Unchanged.

## Stool pair: stool removed (2026-10-13)

`data/Final_{A,B}.mp4`, a room corner with a stool (removed in B), a bag and a ball (both
unchanged). Logs `out/logs/final_run{,2}.log`.

- First run: 154/154 registered, 1.67 px, but 0 detections. Aligned depth maps were flat
  (per-frame p95/p5 depth ratio 1.05-1.09; static 1.35-1.52, bottle 1.82-1.98), because the
  sparse depths per frame were (p95/p5 1.14 vs 1.67-1.78). Triangulation angles were normal
  (median max 4.6 deg; static 3.7, bottle 4.8), so the camera did move enough.
- Cause: self-calibration. Solved SIMPLE_RADIAL focal / width and k per pair: bottle
  1.631 / 0.034, static 1.646 / 0.049, keyboard 1.670 / 0.072, walkway 1.541 / 0.005; the
  failed ball pair 2.342 / 0.189 and this pair 2.209 / 0.429. Same phone and video mode,
  so the long focal plus strong distortion is a degenerate solution that compresses depth.
- Fix: `sfm.known_focal_rel` (or `--known_focal_rel`) sets the focal length as a prior and
  keeps it fixed (`ba_refine_focal_length = False`); distortion is still refined. 1.64 is the
  median of the four self-calibrated pairs. Default stays null, so earlier runs are
  unchanged. `sfm_report.json` now flags |k| > 0.15 as suspicious.
- Second run with 1.64: k = 0.034, 154/154, 1.66 px, depth fits 80/80 and 74/74. Detections:
  stool removed (VLM "footstool" 0.9), 0.57 x 0.44 x 0.42 m, 5 view groups; a second
  removed cluster on the stool's left side (split by DBSCAN); 2 false added, the larger
  (1.88 x 0.73 x 1.14 m) on the blank white wall left of the stool, where B's surface is
  about 13 cm in front of A's; one 0.20 m fragment under the stool. Baseline: 2 moved,
  6 removed, 2 added.
- Heights on this pair are not reliable: the plane RANSAC picks is real floor near the
  camera (its inliers sit 1.23-1.51 m below the cameras along the estimated up) but tilted
  22 deg from that up; the phone was visibly rolled, which the x-axis up estimate assumes
  away. Camera height 0.88 m, so floor snapping and the route check stayed off.

## Cleanup of unused items (2026-10-14)

- Ball pair (`New_change_A/B`) is no longer processed; it stays in the README only as the
  glossy-floor / screen limitation. Its outputs `out/new_change/` (437 MB) were deleted; the
  recordings and `out/logs/new_change_run.log` are kept as evidence.
- Deleted two Hugging Face models nothing references (google/siglip-base-patch16-224,
  openai/clip-vit-base-patch32; about 2 GB) and the Python/pytest/ruff caches.
- Checked the code: every function is referenced and every config key is read, so no code
  was removed.

## SigLIP label check (2026-10-14)

Added an optional check after the VLM (`semantics.label_check`, now on by default). SigLIP
scores the VLM label plus the 80 COCO class names (a fixed public list, not picked for these
recordings) on a tight crop of the view with the object and of the view without it. The
score difference cancels the shared background. Verdicts: `no_visible_object` (no name gains
>= `min_object_evidence` 0.01), `label_contradicted` (the VLM label fits the empty view
better), else `label_supported`. It adds `label_check` to labels.json and a table to
report.md; labels and detections are not changed. Moved changes are not checked.
New deps: sentencepiece, protobuf (SigLIP tokenizer). Test: `test_label_verdict_*`; 50 passed.

Truth from the crops: bottle #0 black tumbler added; stool #0 stool removed, #1 other half of
the stool removed, #2 false (dark floor under the stool), #3 false (blank wall).

| change | VLM | base: verdict (VLM gain; best) | so400m: VLM gain; best |
|---|---|---|---|
| bottle 0 | cup | supported (+0.042; cup, bottle +0.027) | +0.121; cup, bottle +0.004 |
| stool 0 | footstool | supported (+0.808) | +0.906 |
| stool 1 | wall | **contradicted** (-0.083; couch +0.073) | -0.554; couch +0.128 |
| stool 2 (false) | desk | **no_visible_object** (max +0.0028) | max +0.0050 |
| stool 3 (false) | stool | **no_visible_object** (max +0.0003) | max +0.0052 |

- Both false changes flagged, no real change flagged; both wrong labels on real changes
  that could be checked ("wall") flagged. Max gain: real 0.042-0.808, false <= 0.005.
- It does not improve naming: both models prefer "cup" to "bottle" on the tumbler, and
  "couch" for the stool half is not right either. The VLM label is never replaced.
- The 0.01 threshold was chosen after seeing these 5 changes; it is not validated on
  held-out data. Not used to drop detections for that reason.
- Base model chosen (0.8 GB; so400m 3.5 GB gives the same verdicts with wider margins).
  so400m removed from the cache again.

### Held-out check on the walkway pair — did not hold; turned off by default

`run.py --out out/multi --stages semantics,viz` (log `out/logs/multi_labelcheck.log`).
Truth from the crops: #0 bag moved (real); #1 "removed" is the spot where B shows the ADDED
bottle (A shows the empty corner), so the detection is wrong; #2, #3, #4 are false (static
vase, window area).

| change | truth | VLM | check |
|---|---|---|---|
| 0 moved | bag moved | backpack | not_checked |
| 1 removed | wrong (bottle was added here) | vase | supported (vase +0.082) |
| 2 added | false, vase static | vase | supported (vase +0.177) |
| 3 added | false, window + vase top | vase | supported (vase +0.333) |
| 4 added | false, dark window | trash can | contradicted (best cup +0.039) |

- 0 of 3 false changes reached `no_visible_object`; the max gains (0.04-0.33) overlap the
  real changes on the other pairs (0.04-0.81), so no threshold separates them.
- Cause: the method assumes the with/without views frame the same place the same way. Here
  the two views come from different viewpoints and the static vase falls inside one tight
  crop and not the other, so "vase" gains. On the stool pair the background was plain wall.
- "label_supported" on a false change is misleading in a report, so `enabled: false` again.
  Code, test and config kept for anyone who wants the flags.

## Shippable examples, 3-D overview, ROS maps (2026-10-14)

- Example inputs: the bottle, stool and walkway pairs re-encoded for the repository
  (same 1080 x 1920 resolution, 10 fps, H.264 CRF 23, no audio, rotation applied to the
  pixels): 12-18 MB per video, `examples/<pair>/input/`. The pipeline samples 2 fps, so
  10 fps loses nothing. Each example was re-run from these files end to end
  (`out/ex_<pair>`, logs `out/logs/ex_<pair>.log`); `examples/<pair>/` holds that run's outputs.
- Bottle from the re-encoded videos: 127/127 registered, 0.969 px, scale 0.0353 m/unit
  (43 frames), 1 added at (-0.18, 1.99, 0.19), 4 view groups; label now "water bottle"
  (0.8). Baseline 6 false. Same detection as the original-video run.
- New viz output `scene_3d.png/.gif`: fused A and B rendered from one elevated virtual
  camera (z-buffered point splats, one voxel wide at their depth), change boxes and camera
  paths drawn in; the gif orbits +-30 deg. The focal length and principal point are fitted
  so 98 % of B's points fill the frame. Test: nearest point wins, points behind the camera
  are dropped.
- New nav output: both occupancy grids written as ROS `map_server` maps (trinary PGM +
  YAML, 0 occupied / 254 free / 205 unknown as map_saver writes them; free = floor
  reconstructed and nothing in the obstacle band). Walkway: 142 x 70 cells at 5 cm; map B
  shows the bag across the corridor where map A has it at its old place. Test checks the
  image orientation, values and origin.
- Box labels in projected views now have a dark outline.
- `pytest` -> 52 passed.
- Stool from the re-encoded videos (`--known_focal_rel 1.64`): 153/153 registered, 1.66 px,
  f = 1476 px, k = 0.030. **Stool removed** as one cluster ("footstool", 0.54 x 0.57 x 0.46,
  5 view groups) and 1 false added (blank wall, labelled "green ball" because the unmoved
  ball is in the crop). Earlier run: the stool split in two and 2 false added. Baseline: 5
  clusters, 4 of them pieces of the stool.
- Walkway from the re-encoded videos, first run: **floor not found**. Same plane as before
  (7.9 deg from camera-up, cameras 1.28 m above it) but holding 2.97 % of the sparse points
  against the 3 % cutoff (3.9 % in the earlier run). Without the floor, no floor snapping
  and no route check, and the bag came out as separate removed + added clusters. The
  cutoff was too close to the values this floor produces: `floor_min_inliers_ratio`
  0.03 -> 0.02. Checked on every run: static and keyboard have no candidate plane at all,
  stool 4.7-5.6 %, bottle and synthetic 47-56 %, so only the walkway pair is affected.
- Walkway re-run from align: **bag moved 1.13 m** (13 view groups); **the ball's old spot
  found** as removed (A shows the ball, B the empty floor; missed in the earlier run); 4
  false at the window end (static vase reported as moved and as added, dark window
  twice). Baseline: 20 clusters, 7 near real changes.
- **Route check planned through unobserved space.** The replanned route on the walkway ran
  through cells neither recording saw: A* treated every non-occupied cell as free. Now the
  new route may only use known-traversable cells: floor reconstructed in either recording
  (grown by one cell) or within the robot radius of either walked path (the floor under
  the walker is rarely in view: only 52-72 % of route cells have floor points on the
  walkway and synthetic runs). Test: with the far side of the corridor unobserved, the
  detour goes the other way; with only the blocked corridor known, there is no route.
- **The route check is not informative on the walkway pair.** The corridor between bed and
  wardrobe reconstructs narrower than a 0.5 m wide robot: 107 of the route's cells are
  already blocked in A's own map, and only the remaining few can count as newly blocked.
  The shipped run reports 6 cells near the start, not attributed to any change; the
  earlier run reported 11 attributed to the bag. A pair meant for the route check needs a
  path at least 1 m wide.
- Example stills now use the before/after views the semantic stage chose for the first
  change (best unoccluded view of the object and of the same place in the other recording)
  instead of the nearest-pose pairing of the side-by-side video, which gave a removed
  object's frame without the object in it.
- `pytest` -> 53 passed.
- 3-D overview quality: the first gif showed square splat blocks, rows of pinholes on walls
  oblique to the voxel grid (missing surface points, background showing through) and
  speckle from per-frame GIF palettes. Now: splats rendered at 2x and averaged down,
  enclosed holes up to 9 px (at 2x) inpainted without growing the silhouette, one shared
  palette for all frames and no dithering, labels on a dark plate ("object" when the VLM
  gave no valid label). Gifs 1.5-2.7 MB. All example visuals regenerated.
- report.md now ends with a "Visuals" section embedding the outputs next to it (3-D
  overview, floor map, links to the side-by-side video and the Rerun file). Each
  `examples/<pair>/` folder has a README.md (the report plus the visuals that ship there,
  the browser viewer link and the input videos), so opening the folder on GitHub shows it.
