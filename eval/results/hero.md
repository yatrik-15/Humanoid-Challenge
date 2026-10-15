# Evaluation: hero

match radius 0.3; GT frame: {"frame": "pipeline"}

| method                  |   #pred |   #GT |   TP |     P |     R |   centroid err |   region P |   region R |   region err |
|-------------------------|---------|-------|------|-------|-------|----------------|------------|------------|--------------|
| visibility-aware (ours) |       2 |     3 |    2 | 1     | 0.667 |          0.151 |      1     |       0.75 |        0.121 |
| naive NN baseline       |       9 |     3 |    2 | 0.222 | 0.667 |          0.166 |      0.364 |       1    |        0.146 |
