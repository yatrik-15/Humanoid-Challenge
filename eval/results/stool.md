# Evaluation: stool

match radius 0.3; GT frame: {"frame": "pipeline"}

| method                  |   #pred |   #GT |   TP |     P |   R |   centroid err |   region P |   region R |   region err |
|-------------------------|---------|-------|------|-------|-----|----------------|------------|------------|--------------|
| visibility-aware (ours) |       1 |     1 |    1 | 1     |   1 |          0.223 |      1     |          1 |        0.223 |
| naive NN baseline       |       7 |     1 |    1 | 0.143 |   1 |          0.288 |      0.143 |          1 |        0.288 |
