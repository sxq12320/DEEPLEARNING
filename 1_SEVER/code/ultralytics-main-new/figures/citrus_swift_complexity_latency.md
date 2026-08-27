# CitrusSwift complexity and measured latency

Device: `cpu`; input: `640`; batch: `1`; warm-up: `10`; iterations: `30`; CPU threads: `1`.

| Model | Params | GFLOPs | Pretrain coverage | Fused median ms | vs ref | Fused p90 ms |
|---|---:|---:|---:|---:|---:|---:|
| 00_reference | 2,842,803 | 10.356 | 98.42% | 153.78 | +0.0% | 154.93 |
| 01_repcontext_backbone | 2,858,931 | 10.369 | 97.83% | 155.90 | +1.4% | 158.01 |
| 02_lska_backbone | 2,916,019 | 10.413 | 95.96% | 156.33 | +1.7% | 158.85 |
| 03_train_aux_head | 2,915,526 | 10.356 | 95.97% | 153.24 | -0.4% | 154.78 |
| 04_lite_head | 2,747,302 | 9.440 | 96.04% | 142.27 | -7.5% | 144.11 |
| 05_fpn_only_neck | 2,192,499 | 9.534 | 97.95% | 143.55 | -6.7% | 145.03 |
| 06_asym_pan_neck | 2,316,211 | 9.933 | 98.06% | 149.94 | -2.5% | 152.24 |
| 07_lska_asym_pan | 2,389,427 | 9.990 | 95.07% | 151.06 | -1.8% | 152.90 |
| 08_citrus_swift_full | 2,293,926 | 9.074 | 92.08% | 139.63 | -9.2% | 141.09 |
| 09_dense_topology_control | 2,930,707 | 10.783 | 95.47% | 178.21 | +15.9% | 179.36 |

These timings are device-specific engineering measurements, not accuracy results. Re-run this script on the deployment GPU and exported TensorRT engine before making a speed claim.
