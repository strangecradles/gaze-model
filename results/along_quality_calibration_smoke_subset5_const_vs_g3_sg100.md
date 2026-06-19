# Along-Quality Calibration

Generated: 2026-06-17T23:02:31

Hard gates per subject: r_dot_x >= baseline - 0.02; valid_frac >= baseline - 0.03; prec_x <= baseline + 0.10; j30 <= baseline + 0.05.

Primary improvement targets: median raw >=3 px jump fraction and median p99.9 raw line-step speed should each fall by at least 10%.

## Config Summary

| stage | variant_id | config_id | slew_gate | slew_max_deg_s | n_subjects | pass_all | median_score | median_raw_jump_reduction | median_raw_speed_p999_reduction | median_delta_prec_x | median_delta_j30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prune | qv_power_s2_6_g3_sg100 | qv_power_s2_6_g3 | yes | 100 | 4 | no | -9.458 | 0.341 | 0.231 | 0.654 | 0.12 |
| prune | constant_sg100 | constant | yes | 100 | 4 | no | -9.615 | 0.313 | 0.136 | -0.047 | 0.263 |

## Subject Rows

| stage | subject | variant_id | config_id | slew_gate | slew_max_deg_s | pass_hard_gates | raw_jump_reduction | raw_speed_p999_reduction | delta_r_dot_x | delta_valid_frac | delta_prec_x | delta_j30 | oculo_j30_reduction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prune | Ashton3 | constant_sg100 | constant | yes | 100 | no | 0.265 | 0.132 | -2.00e-03 | -4.05e-03 | 1.33e-03 | 0.08 | -0.258 |
| prune | Ashton3 | qv_power_s2_6_g3_sg100 | qv_power_s2_6_g3 | yes | 100 | no | 0.353 | 0.232 | -1.01e-03 | 2.05e-03 | 1.086 | 0.103 | -0.127 |
| prune | Chong | constant_sg100 | constant | yes | 100 | no | 0.297 | 0.127 | -1.48e-03 | -5.85e-03 | -0.096 | 0.194 | -0.032 |
| prune | Chong | qv_power_s2_6_g3_sg100 | qv_power_s2_6_g3 | yes | 100 | no | 0.298 | 0.229 | -2.12e-03 | 3.25e-03 | 0.283 | 0.011 | 0.018 |
| prune | Kathy | constant_sg100 | constant | yes | 100 | no | 0.328 | 0.199 | -1.55e-03 | -0.011 | -0.121 | 0.376 | -0.093 |
| prune | Kathy | qv_power_s2_6_g3_sg100 | qv_power_s2_6_g3 | yes | 100 | no | 0.352 | 0.275 | -2.54e-03 | 3.79e-03 | 0.826 | 0.137 | 0.223 |
| prune | Pavel | constant_sg100 | constant | yes | 100 | no | 0.333 | 0.14 | -7.26e-04 | -0.016 | 0.102 | 0.331 | -0.479 |
| prune | Pavel | qv_power_s2_6_g3_sg100 | qv_power_s2_6_g3 | yes | 100 | no | 0.33 | 0.212 | -1.86e-03 | -8.00e-04 | 0.482 | 0.163 | -0.223 |

## Leave-One-Subject-Out

| holdout | selected_config | pass_hard_gates | raw_jump_reduction | raw_speed_p999_reduction | delta_prec_x | delta_j30 |
| --- | --- | --- | --- | --- | --- | --- |
| Ashton3 | qv_power_s2_6_g3_sg100 | no | 0.353 | 0.232 | 1.086 | 0.103 |
| Chong | qv_power_s2_6_g3_sg100 | no | 0.298 | 0.229 | 0.283 | 0.011 |
| Kathy | qv_power_s2_6_g3_sg100 | no | 0.352 | 0.275 | 0.826 | 0.137 |
| Pavel | qv_power_s2_6_g3_sg100 | no | 0.33 | 0.212 | 0.482 | 0.163 |
