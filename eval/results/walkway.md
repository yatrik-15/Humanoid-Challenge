# Evaluation: walkway

match radius 0.3; GT frame: {"frame": "pipeline"}

| method                  |   #pred |   #GT |   TP |    P |     R |   centroid err |   region P |   region R |   region err |
|-------------------------|---------|-------|------|------|-------|----------------|------------|------------|--------------|
| visibility-aware (ours) |       4 |     3 |    1 | 0.25 | 0.333 |          0.277 |      0.6   |        0.6 |        0.199 |
| naive NN baseline       |      10 |     3 |    2 | 0.2  | 0.667 |          0.178 |      0.364 |        0.8 |        0.178 |
