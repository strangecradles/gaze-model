# Along-Quality Calibration

Generated: 2026-06-17T23:05:18

Hard gates per subject: r_dot_x >= baseline - 0.02; valid_frac >= baseline - 0.03; prec_x <= baseline + 0.10; j30 <= baseline + 0.05.

Primary improvement targets: median raw >=3 px jump fraction and median p99.9 raw line-step speed should each fall by at least 10%.

## Config Summary

| stage | variant_id | config_id | slew_gate | slew_max_deg_s | n_subjects | pass_all | median_score | median_raw_jump_reduction | median_raw_speed_p999_reduction | median_delta_prec_x | median_delta_j30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prune | constant | constant | no |  | 4 | no | -9.629 | 0.284 | 0.148 | -0.058 | 0.231 |

## Subject Rows

| stage | subject | variant_id | config_id | slew_gate | slew_max_deg_s | pass_hard_gates | raw_jump_reduction | raw_speed_p999_reduction | delta_r_dot_x | delta_valid_frac | delta_prec_x | delta_j30 | oculo_j30_reduction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prune | Ashton3 | constant | constant | no |  | no | 0.243 | 0.12 | -1.94e-03 | -4.05e-03 | -0.017 | 0.082 | -0.244 |
| prune | Chong | constant | constant | no |  | no | 0.272 | 0.117 | -1.52e-03 | -5.85e-03 | -0.1 | 0.193 | -0.032 |
| prune | Kathy | constant | constant | no |  | no | 0.295 | 0.213 | -1.46e-03 | -0.011 | -0.148 | 0.269 | -0.175 |
| prune | Pavel | constant | constant | no |  | no | 0.302 | 0.176 | -6.08e-04 | -0.016 | 0.101 | 0.358 | -0.467 |
