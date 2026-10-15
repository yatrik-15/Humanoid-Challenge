# Evaluation: hero

match radius 0.3; GT frame: {"frame": "pipeline"}

| method                  |   #pred |   #GT |   TP |   P |   R |   centroid err |   region P |   region R |   region err |
|-------------------------|---------|-------|------|-----|-----|----------------|------------|------------|--------------|
| visibility-aware (ours) |       3 |     3 |    3 | 1   |   1 |          0.187 |      1     |          1 |        0.156 |
| naive NN baseline       |       5 |     3 |    3 | 0.6 |   1 |          0.165 |      0.571 |          1 |        0.124 |
