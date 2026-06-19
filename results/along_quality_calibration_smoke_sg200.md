# Along-Quality Calibration

Generated: 2026-06-17T22:47:04

Hard gates per subject: r_dot_x >= baseline - 0.02; valid_frac >= baseline - 0.03; prec_x <= baseline + 0.10; j30 <= baseline + 0.05.

Primary improvement targets: median raw >=3 px jump fraction and median p99.9 raw line-step speed should each fall by at least 10%.

## Config Summary

| stage | config_id | n_subjects | pass_all | median_score | median_raw_jump_reduction | median_raw_speed_p999_reduction | median_delta_prec_x | median_delta_j30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prune | qv_power_s2_6_g0p5 | 1 | no | -9.651 | 0.146 | 0.203 | 0.144 | 1.42e-14 |

## Subject Rows

| stage | subject | config_id | pass_hard_gates | raw_jump_reduction | raw_speed_p999_reduction | delta_r_dot_x | delta_valid_frac | delta_prec_x | delta_j30 | oculo_j30_reduction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prune | Ashton3 | qv_power_s2_6_g0p5 | no | 0.146 | 0.203 | nan | -1.44e-03 | 0.144 | 1.42e-14 | 0 |
