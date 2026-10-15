# What changed in the room

Units: **approx. metres** (scale from monocular metric depth (median over frames)). World frame: +z up; z=0 is the dominant horizontal support plane found by RANSAC (the floor, or a bed/table top when the floor is not visible).

Detected: **1 moved, 1 removed, 2 added**. Regions observed in only one recording are reported as UNKNOWN (2557 evidence voxels), never as changes.

| id | action | object | label conf. | where (centroid x, y, z) | size / displacement | view groups |
|---|---|---|---|---|---|---|
| 0 | moved | backpack | 0.80 | (0.00, 0.08, 0.22) → (0.03, 1.16, 0.27) | moved 1.08 | 13 |
| 1 | removed | (no label) | 0.00 | (0.01, 2.29, 0.16) | 0.22 x 0.16 x 0.15 | 32 |
| 2 | added | vase | 0.80 | (0.92, 2.97, 0.47) | 0.35 x 0.29 x 0.78 | 31 |
| 3 | added | desk | 0.80 | (0.24, 3.43, 0.20) | 0.20 x 0.08 x 0.05 | 18 |

## Descriptions

- **#0 moved — backpack**: Backpack moved: An object moved. Before: centre at x=0.00, y=0.08 m (approx.); size about 0.55 x 0.38 x 0.35 m (approx.) (top 0.39 m (approx.) above the z=0 plane). After: centre at x=0.03, y=1.16 m (approx.); size about 0.50 x 0.41 x 0.42 m (approx.) (top 0.48 m (approx.) above the z=0 plane). Displacement 1.08 m (approx.) (dx=+0.03, dy=+1.08, dz=+0.05). The backpack is now placed on the floor, near the bed.
- **#1 removed — (no label)**: An object was removed (present only in the before recording). Region: centre at x=0.01, y=2.29 m (approx.); size about 0.22 x 0.16 x 0.15 m (approx.) (top 0.23 m (approx.) above the z=0 plane). (no valid semantic label)
- **#2 added — vase**: Vase added: An object was added (present only in the after recording). Region: centre at x=0.92, y=2.97 m (approx.); size about 0.35 x 0.29 x 0.78 m (approx.) (top 0.87 m (approx.) above the z=0 plane). The vase is on the floor, near the window, and is in front of the window.
- **#3 added — desk**: Desk added: An object was added (present only in the after recording). Region: centre at x=0.24, y=3.43 m (approx.); size about 0.20 x 0.08 x 0.05 m (approx.) (top 0.23 m (approx.) above the z=0 plane). The desk is located in the middle of the room, between the two visible walls.
