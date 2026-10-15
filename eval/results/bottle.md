# Evaluation: bottle

match radius 0.3; GT frame: {"frame": "pipeline"}

| method                  |   #pred |   #GT |   TP |     P |   R |   centroid err |   region P |   region R |   region err |
|-------------------------|---------|-------|------|-------|-----|----------------|------------|------------|--------------|
| visibility-aware (ours) |       1 |     1 |    1 | 1     |   1 |          0.039 |      1     |          1 |        0.039 |
| naive NN baseline       |       3 |     1 |    1 | 0.333 |   1 |          0.035 |      0.333 |          1 |        0.035 |
