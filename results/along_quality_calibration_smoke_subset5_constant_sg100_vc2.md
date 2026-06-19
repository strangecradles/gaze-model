# Along-Quality Calibration

Generated: 2026-06-17T23:09:15

Hard gates per subject: r_dot_x >= baseline - 0.02; valid_frac >= baseline - 0.03; prec_x <= baseline + 0.10; j30 <= baseline + 0.05.

Primary improvement targets: median raw >=3 px jump fraction and median p99.9 raw line-step speed should each fall by at least 10%.

## Config Summary

| stage | variant_id | config_id | slew_gate | slew_max_deg_s | hypothesis_velocity_cost | n_subjects | pass_all | median_score | median_raw_jump_reduction | median_raw_speed_p999_reduction | median_delta_prec_x | median_delta_j30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prune | constant_sg100_vc2 | constant | yes | 100 | 2 | 2 | no | -9.614 | 0.296 | 0.129 | -0.045 | 0.136 |

## Subject Rows

| stage | subject | variant_id | config_id | slew_gate | slew_max_deg_s | hypothesis_velocity_cost | pass_hard_gates | raw_jump_reduction | raw_speed_p999_reduction | delta_r_dot_x | delta_valid_frac | delta_prec_x | delta_j30 | oculo_j30_reduction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prune | Ashton3 | constant_sg100_vc2 | constant | yes | 100 | 2 | no | 0.277 | 0.132 | -2.02e-03 | -4.05e-03 | 2.17e-03 | 0.08 | -0.258 |
| prune | Chong | constant_sg100_vc2 | constant | yes | 100 | 2 | no | 0.314 | 0.127 | -1.47e-03 | -5.85e-03 | -0.092 | 0.191 | -0.056 |
