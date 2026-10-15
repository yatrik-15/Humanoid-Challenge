# What changed in the room

Units: **approx. metres** (scale from monocular metric depth (median over frames)). World frame: +z up; z=0 is the dominant horizontal support plane found by RANSAC (the floor, or a bed/table top when the floor is not visible).

Detected: **0 moved, 1 removed, 0 added**. Regions observed in only one recording are reported as UNKNOWN (1537 evidence voxels), never as changes.

| id | action | object | label conf. | where (centroid x, y, z) | size / displacement | view groups |
|---|---|---|---|---|---|---|
| 0 | removed | footstool | 0.90 | (-0.59, 2.85, 0.96) | 0.54 x 0.53 x 0.44 | 5 |

## Descriptions

- **#0 removed — footstool**: Footstool removed: An object was removed (present only in the before recording). Region: centre at x=-0.59, y=2.85 m (approx.); size about 0.54 x 0.53 x 0.44 m (approx.) (top 1.18 m (approx.) above the z=0 plane). The footstool is located on the floor, adjacent to the wall and the corner of the room.

## Visuals

Both recordings fused in 3-D, seen from one virtual camera (left before, right after):

![3-D overview](scene_3d.gif)

Best before / after view of the first change, with every change box projected in:

![before and after](side_by_side.jpg)

Top-down map (filled = there now, dashed = gone or old position):

<img src="floor_map.png" width="520" alt="top-down change map">

Interactive 3-D in the browser: [open in the Rerun web viewer](https://app.rerun.io/version/0.39.0/?url=https://raw.githubusercontent.com/yatrik-15/Humanoid-Challenge/main/examples/stool/changes.rrd), or locally `rerun changes.rrd`.

Input videos: [`input/A.mp4`](input/A.mp4) (before), [`input/B.mp4`](input/B.mp4) (after).
