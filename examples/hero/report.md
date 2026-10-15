# What changed in the room

Units: **approx. metres** (scale from monocular metric depth (median over frames)). World frame: +z up; z=0 is the dominant horizontal support plane found by RANSAC (the floor, or a bed/table top when the floor is not visible).

Detected: **1 moved, 0 removed, 1 added**. Regions observed in only one recording are reported as UNKNOWN (860 evidence voxels), never as changes.

| id | action | object | label conf. | where (centroid x, y, z) | size / displacement | view groups |
|---|---|---|---|---|---|---|
| 0 | moved | backpack | 0.80 | (0.06, 1.43, 0.12) → (-1.03, 1.81, 0.47) | moved 1.20 | 13 |
| 1 | added | stool | 0.80 | (0.42, 1.53, 0.26) | 0.63 x 0.43 x 0.42 | 8 |

## Descriptions

- **#0 moved — backpack**: Backpack moved: An object moved. Before: centre at x=0.06, y=1.43 m (approx.); size about 0.31 x 0.19 x 0.39 m (approx.) (top 0.32 m (approx.) above the z=0 plane). After: centre at x=-1.03, y=1.81 m (approx.); size about 0.48 x 0.20 x 0.30 m (approx.) (top 0.61 m (approx.) above the z=0 plane). Displacement 1.20 m (approx.) (dx=-1.09, dy=+0.38, dz=+0.34). The backpack is on top of the bed, which is on top of the floor.
- **#1 added — stool**: Stool added: An object was added (present only in the after recording). Region: centre at x=0.42, y=1.53 m (approx.); size about 0.63 x 0.43 x 0.42 m (approx.) (top 0.47 m (approx.) above the z=0 plane). The stool is located near the bed, on the floor, and is adjacent to the bed frame.
