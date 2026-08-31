# CitrusSwift complexity and measured latency

Device: `cpu`; input: `640`; batch: `1`; warm-up: `10`; iterations: `30`; CPU threads: `1`.

| Model | Params | GFLOPs | Pretrain coverage | Fused median ms | vs ref | Fused p90 ms |
|---|---:|---:|---:|---:|---:|---:|
| 00_reference | 2,842,803 | 10.356 | 98.42% | 150.73 | +0.0% | 155.07 |
| 01_repcontext_backbone | 2,858,931 | 10.369 | 97.83% | 153.45 | +1.8% | 158.24 |
| 02_lska_backbone | 2,916,019 | 10.413 | 95.96% | 155.33 | +3.0% | 159.04 |
| 03_train_aux_head | 2,915,526 | 10.356 | 95.97% | 153.28 | +1.7% | 156.46 |
| 04_lite_head | 2,747,302 | 9.440 | 96.04% | 141.50 | -6.1% | 146.42 |
| 05_fpn_only_neck | 2,192,499 | 9.534 | 97.95% | 143.15 | -5.0% | 146.72 |
| 06_asym_pan_neck | 2,316,211 | 9.933 | 98.06% | 151.75 | +0.7% | 155.39 |
| 07_lska_asym_pan | 2,389,427 | 9.990 | 95.07% | 150.79 | +0.0% | 154.75 |
| 08_citrus_swift_full | 2,293,926 | 9.074 | 92.08% | 137.63 | -8.7% | 147.42 |
| 09_dense_topology_control | 2,930,707 | 10.783 | 95.47% | 180.80 | +19.9% | 187.14 |

These timings are device-specific engineering measurements, not accuracy results. Re-run this script on the deployment GPU and exported TensorRT engine before making a speed claim.
