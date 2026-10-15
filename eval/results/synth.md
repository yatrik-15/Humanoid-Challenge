# Evaluation: synth

match radius 0.3; GT frame: {"frame": "gt_world mapped by Umeyama on camera centres", "num_cameras": 60, "scale": 1.016903969333803, "camera_residual_mean": 0.001440473878802089, "camera_residual_max": 0.002849535567306297}

| method                  |   #pred |   #GT |   TP |     P |   R |   centroid err |   region P |   region R |   region err |
|-------------------------|---------|-------|------|-------|-----|----------------|------------|------------|--------------|
| visibility-aware (ours) |       3 |     3 |    3 | 1     |   1 |          0.099 |      1     |          1 |        0.075 |
| naive NN baseline       |      13 |     3 |    3 | 0.231 |   1 |          0.109 |      0.286 |          1 |        0.085 |
