# What changed in the room

Units: **approx. metres** (scale from monocular metric depth (median over frames)). World frame: +z up; z=0 is the dominant horizontal support plane found by RANSAC (the floor, or a bed/table top when the floor is not visible).

Detected: **1 moved, 1 removed, 1 added**. Regions observed in only one recording are reported as UNKNOWN (958 evidence voxels), never as changes.

| id | action | object | label conf. | where (centroid x, y, z) | size / displacement | view groups |
|---|---|---|---|---|---|---|
| 0 | moved | backpack | 0.80 | (0.55, 1.20, -0.12) → (1.00, 1.69, 0.38) | moved 0.83 | 8 |
| 1 | removed | (no label) | 0.00 | (0.38, 1.99, 0.49) | 0.29 x 0.21 x 0.23 | 10 |
| 2 | added | stool | 0.80 | (0.99, 0.84, 0.08) | 0.59 x 0.57 x 0.95 | 5 |

## Descriptions

- **#0 moved — backpack**: Backpack moved: An object moved. Before: centre at x=0.55, y=1.20 m (approx.); size about 0.30 x 0.20 x 0.36 m (approx.) (top 0.06 m (approx.) above the z=0 plane). After: centre at x=1.00, y=1.69 m (approx.); size about 0.52 x 0.31 x 0.30 m (approx.) (top 0.53 m (approx.) above the z=0 plane). Displacement 0.83 m (approx.) (dx=+0.45, dy=+0.49, dz=+0.49). The backpack is on the bed, with the red blanket behind it and the door to the right.
- **#1 removed — (no label)**: An object was removed (present only in the before recording). Region: centre at x=0.38, y=1.99 m (approx.); size about 0.29 x 0.21 x 0.23 m (approx.) (top 0.61 m (approx.) above the z=0 plane). (no valid semantic label)
- **#2 added — stool**: Stool added: An object was added (present only in the after recording). Region: centre at x=0.99, y=0.84 m (approx.); size about 0.59 x 0.57 x 0.95 m (approx.) (top 0.56 m (approx.) above the z=0 plane). The stool is on the floor near the bed and the door, close to the wall.
