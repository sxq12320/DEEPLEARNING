# E V5->V6 evidence table (auto-generated; verify against raw JSONs)

## Official validation (per-run best_mask checkpoint)

| run | series | mask AP50-95 | mask AP50 | best epoch | final AP50-95 |
|---|---|---:|---:|---:|---:|
| V5_00_control | V5 | 0.67653 | 0.83891 | 220 | 0.67017 |
| V5_01_fine | V5 | 0.68125 | 0.84266 | 175 | 0.66890 |
| V5_02_mask_route | V5 | 0.67641 | 0.83639 | 210 | 0.66907 |
| V5_03_region | V5 | 0.67911 | 0.83664 | 254 | 0.66737 |
| V5_04_fine_route | V5 | 0.67754 | 0.84228 | 241 | 0.66685 |
| V5_07_full | V5 | 0.67604 | 0.83892 | 277 | 0.66612 |

## Factor wiring (V6)

| run | fine_mask | nwd_ratio | copy_paste | mask_ratio |
|---|---|---|---|---|
| V6_00_control | False | 0.0 | 0.0 | 4 |
| V6_01_mr2 | False | 0.0 | 0.0 | 2 |
| V6_02_fine | True | 0.0 | 0.0 | 2 |
| V6_03_nwd | False | 0.5 | 0.0 | 4 |
| V6_04_cp | False | 0.0 | 0.3 | 4 |
| V6_05_fine_nwd | True | 0.5 | 0.0 | 2 |
| V6_06_fine_cp | True | 0.0 | 0.3 | 2 |
| V6_07_full | True | 0.5 | 0.3 | 2 |