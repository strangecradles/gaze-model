# Along-Quality Calibration

Generated: 2026-06-17T22:51:49

Hard gates per subject: r_dot_x >= baseline - 0.02; valid_frac >= baseline - 0.03; prec_x <= baseline + 0.10; j30 <= baseline + 0.05.

Primary improvement targets: median raw >=3 px jump fraction and median p99.9 raw line-step speed should each fall by at least 10%.

## Config Summary

| stage | variant_id | config_id | slew_gate | slew_max_deg_s | n_subjects | pass_all | median_score | median_raw_jump_reduction | median_raw_speed_p999_reduction | median_delta_prec_x | median_delta_j30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prune | qv_power_s2_6_g0p5_sg100 | qv_power_s2_6_g0p5 | yes | 100 | 7 | no | -9.318 | 0.324 | 0.237 | 0.239 | 0 |
| prune | qv_power_s2_6_g0p5_sg75 | qv_power_s2_6_g0p5 | yes | 75 | 7 | no | -9.438 | 0.319 | 0.241 | 0.243 | 0 |

## Subject Rows

| stage | subject | variant_id | config_id | slew_gate | slew_max_deg_s | pass_hard_gates | raw_jump_reduction | raw_speed_p999_reduction | delta_r_dot_x | delta_valid_frac | delta_prec_x | delta_j30 | oculo_j30_reduction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prune | Ashton3 | qv_power_s2_6_g0p5_sg75 | qv_power_s2_6_g0p5 | yes | 75 | no | 0.109 | 0.175 | nan | -1.44e-03 | 0.109 | 3.55e-15 | 0 |
| prune | Ashton3 | qv_power_s2_6_g0p5_sg100 | qv_power_s2_6_g0p5 | yes | 100 | yes | 0.113 | 0.113 | nan | -1.44e-03 | 0.064 | 7.11e-15 | 0 |
| prune | Chong | qv_power_s2_6_g0p5_sg75 | qv_power_s2_6_g0p5 | yes | 75 | no | 0.321 | 0.241 | nan | 6.31e-03 | 1.156 | 0 | 0 |
| prune | Chong | qv_power_s2_6_g0p5_sg100 | qv_power_s2_6_g0p5 | yes | 100 | no | 0.324 | 0.237 | nan | 6.31e-03 | 1.156 | 0 | 0 |
| prune | Igor | qv_power_s2_6_g0p5_sg75 | qv_power_s2_6_g0p5 | yes | 75 | no | 0.319 | 0.118 | nan | 6.25e-03 | 0.675 | 0 | 0 |
| prune | Igor | qv_power_s2_6_g0p5_sg100 | qv_power_s2_6_g0p5 | yes | 100 | no | 0.335 | 0.191 | nan | 6.25e-03 | 0.671 | 0 | 0 |
| prune | John | qv_power_s2_6_g0p5_sg75 | qv_power_s2_6_g0p5 | yes | 75 | no | 0.239 | 0.257 | nan | -4.14e-04 | 0.243 | 1.42e-14 | 0 |
| prune | John | qv_power_s2_6_g0p5_sg100 | qv_power_s2_6_g0p5 | yes | 100 | no | 0.239 | 0.258 | nan | -4.14e-04 | 0.239 | 7.11e-15 | 0 |
| prune | Kathy | qv_power_s2_6_g0p5_sg75 | qv_power_s2_6_g0p5 | yes | 75 | no | 0.341 | 0.419 | -6.14e-17 | 5.24e-03 | 0.472 | -7.11e-15 | 0 |
| prune | Kathy | qv_power_s2_6_g0p5_sg100 | qv_power_s2_6_g0p5 | yes | 100 | no | 0.357 | 0.325 | 4.07e-19 | 5.24e-03 | 0.517 | -7.11e-15 | 0 |
| prune | Pavel | qv_power_s2_6_g0p5_sg75 | qv_power_s2_6_g0p5 | yes | 75 | yes | 0.344 | 0.451 | -6.10e-17 | -7.84e-04 | -0.351 | 0 | 0 |
| prune | Pavel | qv_power_s2_6_g0p5_sg100 | qv_power_s2_6_g0p5 | yes | 100 | yes | 0.359 | 0.491 | 3.92e-17 | -7.84e-04 | -0.325 | 0 | 0 |
| prune | Zohre | qv_power_s2_6_g0p5_sg75 | qv_power_s2_6_g0p5 | yes | 75 | yes | 0.283 | 0.204 | -4.97e-17 | 4.13e-03 | -1.682 | 1.07e-14 | 0 |
| prune | Zohre | qv_power_s2_6_g0p5_sg100 | qv_power_s2_6_g0p5 | yes | 100 | yes | 0.294 | 0.204 | 7.61e-17 | 4.13e-03 | -1.672 | 0 | 0 |

## Leave-One-Subject-Out

| holdout | selected_config | pass_hard_gates | raw_jump_reduction | raw_speed_p999_reduction | delta_prec_x | delta_j30 |
| --- | --- | --- | --- | --- | --- | --- |
| Ashton3 | qv_power_s2_6_g0p5_sg75 | no | 0.109 | 0.175 | 0.109 | 3.55e-15 |
| Chong | qv_power_s2_6_g0p5_sg100 | no | 0.324 | 0.237 | 1.156 | 0 |
| Igor | qv_power_s2_6_g0p5_sg100 | no | 0.335 | 0.191 | 0.671 | 0 |
| John | qv_power_s2_6_g0p5_sg100 | no | 0.239 | 0.258 | 0.239 | 7.11e-15 |
| Kathy | qv_power_s2_6_g0p5_sg100 | no | 0.357 | 0.325 | 0.517 | -7.11e-15 |
| Pavel | qv_power_s2_6_g0p5_sg100 | yes | 0.359 | 0.491 | -0.325 | 0 |
| Zohre | qv_power_s2_6_g0p5_sg100 | yes | 0.294 | 0.204 | -1.672 | 0 |
