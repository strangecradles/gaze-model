# Along-Quality Calibration

Generated: 2026-06-17T22:57:30

Hard gates per subject: r_dot_x >= baseline - 0.02; valid_frac >= baseline - 0.03; prec_x <= baseline + 0.10; j30 <= baseline + 0.05.

Primary improvement targets: median raw >=3 px jump fraction and median p99.9 raw line-step speed should each fall by at least 10%.

## Config Summary

| stage | variant_id | config_id | slew_gate | slew_max_deg_s | n_subjects | pass_all | median_score | median_raw_jump_reduction | median_raw_speed_p999_reduction | median_delta_prec_x | median_delta_j30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prune | qv_power_s2_6_g3_sg100 | qv_power_s2_6_g3 | yes | 100 | 7 | no | 0.467 | 0.297 | 0.223 | 0.068 | 0 |
| prune | constant_sg100 | constant | yes | 100 | 7 | no | 0.392 | 0.273 | 0.262 | 0.05 | 0 |

## Subject Rows

| stage | subject | variant_id | config_id | slew_gate | slew_max_deg_s | pass_hard_gates | raw_jump_reduction | raw_speed_p999_reduction | delta_r_dot_x | delta_valid_frac | delta_prec_x | delta_j30 | oculo_j30_reduction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prune | Ashton3 | constant_sg100 | constant | yes | 100 | yes | 0.19 | 0.281 | nan | -7.35e-03 | 0.05 | 0 | 0 |
| prune | Ashton3 | qv_power_s2_6_g3_sg100 | qv_power_s2_6_g3 | yes | 100 | no | 0.174 | 0.179 | nan | -1.71e-03 | 0.221 | 0 | 0 |
| prune | Chong | constant_sg100 | constant | yes | 100 | no | 0.27 | -0.219 | nan | -6.41e-03 | -0.065 | 0 | 0 |
| prune | Chong | qv_power_s2_6_g3_sg100 | qv_power_s2_6_g3 | yes | 100 | no | 0.298 | -0.026 | nan | 6.11e-03 | 0.952 | 0 | 0 |
| prune | Igor | constant_sg100 | constant | yes | 100 | yes | 0.305 | 0.087 | nan | -0.012 | -0.289 | 0 | 0 |
| prune | Igor | qv_power_s2_6_g3_sg100 | qv_power_s2_6_g3 | yes | 100 | yes | 0.344 | 0.223 | nan | -8.97e-04 | 0.068 | 0 | 0 |
| prune | John | constant_sg100 | constant | yes | 100 | yes | 0.255 | 0.277 | nan | -7.01e-05 | 0.076 | 7.11e-15 | 0 |
| prune | John | qv_power_s2_6_g3_sg100 | qv_power_s2_6_g3 | yes | 100 | yes | 0.191 | 0.277 | nan | -1.32e-06 | 0.04 | 7.11e-15 | 0 |
| prune | Kathy | constant_sg100 | constant | yes | 100 | no | 0.273 | 0.298 | nan | -5.56e-03 | 0.091 | -7.11e-15 | 0 |
| prune | Kathy | qv_power_s2_6_g3_sg100 | qv_power_s2_6_g3 | yes | 100 | no | 0.288 | 0.422 | -1.23e-16 | 4.14e-03 | 0.197 | -7.11e-15 | 0 |
| prune | Pavel | constant_sg100 | constant | yes | 100 | no | 0.29 | 0.262 | -2.80e-17 | -0.012 | 0.104 | 0 | 0 |
| prune | Pavel | qv_power_s2_6_g3_sg100 | qv_power_s2_6_g3 | yes | 100 | yes | 0.297 | 0.409 | -6.10e-17 | -3.88e-03 | -0.28 | 2.84e-14 | 0 |
| prune | Zohre | constant_sg100 | constant | yes | 100 | yes | 0.359 | 0.237 | 8.98e-17 | -9.69e-03 | -2.553 | 7.11e-15 | 0 |
| prune | Zohre | qv_power_s2_6_g3_sg100 | qv_power_s2_6_g3 | yes | 100 | yes | 0.355 | 0.217 | -2.18e-17 | -6.14e-04 | -2.06 | 1.42e-14 | 0 |

## Leave-One-Subject-Out

| holdout | selected_config | pass_hard_gates | raw_jump_reduction | raw_speed_p999_reduction | delta_prec_x | delta_j30 |
| --- | --- | --- | --- | --- | --- | --- |
| Ashton3 | qv_power_s2_6_g3_sg100 | no | 0.174 | 0.179 | 0.221 | 0 |
| Chong | qv_power_s2_6_g3_sg100 | no | 0.298 | -0.026 | 0.952 | 0 |
| Igor | qv_power_s2_6_g3_sg100 | yes | 0.344 | 0.223 | 0.068 | 0 |
| John | qv_power_s2_6_g3_sg100 | yes | 0.191 | 0.277 | 0.04 | 7.11e-15 |
| Kathy | qv_power_s2_6_g3_sg100 | no | 0.288 | 0.422 | 0.197 | -7.11e-15 |
| Pavel | constant_sg100 | no | 0.29 | 0.262 | 0.104 | 0 |
| Zohre | qv_power_s2_6_g3_sg100 | yes | 0.355 | 0.217 | -2.06 | 1.42e-14 |
