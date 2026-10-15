# Evaluation: keyboard

match radius 0.3; GT frame: {"frame": "pipeline"}

| method                  |   #pred |   #GT |   TP | P   |   R | centroid err   | region P   |   region R | region err   |
|-------------------------|---------|-------|------|-----|-----|----------------|------------|------------|--------------|
| visibility-aware (ours) |       0 |     1 |    0 |     |   0 |                |            |          0 |              |
| naive NN baseline       |       0 |     1 |    0 |     |   0 |                |            |          0 |              |
