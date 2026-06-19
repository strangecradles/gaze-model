# Along-Quality Calibration

Generated: 2026-06-17T23:19:54

Hard gates per subject: r_dot_x >= baseline - 0.02; valid_frac >= baseline - 0.03; prec_x <= baseline + 0.10; j30 <= baseline + 0.05.

Primary improvement targets: median raw >=3 px jump fraction and median p99.9 raw line-step speed should each fall by at least 10%.

## Config Summary

| stage | variant_id | config_id | slew_gate | slew_max_deg_s | hypothesis_velocity_cost | hypothesis_acceleration_cost | n_subjects | pass_all | median_score | median_raw_jump_reduction | median_raw_speed_p999_reduction | median_delta_prec_x | median_delta_j30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prune | constant_sg100_ac1 | constant | yes | 100 | 0 | 1 | 2 | no | -9.556 | 0.208 | 0.266 | -0.07 | 0.131 |

## Subject Rows

| stage | subject | variant_id | config_id | slew_gate | slew_max_deg_s | hypothesis_velocity_cost | hypothesis_acceleration_cost | pass_hard_gates | raw_jump_reduction | raw_speed_p999_reduction | delta_r_dot_x | delta_valid_frac | delta_prec_x | delta_j30 | oculo_j30_reduction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prune | Ashton3 | constant_sg100_ac1 | constant | yes | 100 | 0 | 1 | no | 0.185 | 0.251 | -1.91e-03 | -4.05e-03 | -0.022 | 0.091 | -0.241 |
| prune | Chong | constant_sg100_ac1 | constant | yes | 100 | 0 | 1 | no | 0.231 | 0.282 | -1.57e-03 | -5.85e-03 | -0.118 | 0.172 | -3.83e-03 |
