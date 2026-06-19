# Along-Quality Calibration

Generated: 2026-06-17T22:55:17

Hard gates per subject: r_dot_x >= baseline - 0.02; valid_frac >= baseline - 0.03; prec_x <= baseline + 0.10; j30 <= baseline + 0.05.

Primary improvement targets: median raw >=3 px jump fraction and median p99.9 raw line-step speed should each fall by at least 10%.

## Config Summary

| stage | variant_id | config_id | slew_gate | slew_max_deg_s | n_subjects | pass_all | median_score | median_raw_jump_reduction | median_raw_speed_p999_reduction | median_delta_prec_x | median_delta_j30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prune | qv_power_s2_6_g3_sg100 | qv_power_s2_6_g3 | yes | 100 | 7 | no | 0.467 | 0.297 | 0.223 | 0.068 | 0 |
| prune | qv_power_s2_6_g0p5_sg100 | qv_power_s2_6_g0p5 | yes | 100 | 7 | no | -9.318 | 0.324 | 0.237 | 0.239 | 0 |
| prune | qv_power_s2_6_g2_sg100 | qv_power_s2_6_g2 | yes | 100 | 7 | no | -9.34 | 0.3 | 0.217 | 0.294 | 0 |
| prune | qv_power_s2_6_g1_sg100 | qv_power_s2_6_g1 | yes | 100 | 7 | no | -9.369 | 0.293 | 0.216 | 0.158 | 0 |

## Subject Rows

| stage | subject | variant_id | config_id | slew_gate | slew_max_deg_s | pass_hard_gates | raw_jump_reduction | raw_speed_p999_reduction | delta_r_dot_x | delta_valid_frac | delta_prec_x | delta_j30 | oculo_j30_reduction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prune | Ashton3 | qv_power_s2_6_g0p5_sg100 | qv_power_s2_6_g0p5 | yes | 100 | yes | 0.113 | 0.113 | nan | -1.44e-03 | 0.064 | 7.11e-15 | 0 |
| prune | Ashton3 | qv_power_s2_6_g1_sg100 | qv_power_s2_6_g1 | yes | 100 | no | 0.134 | 0.209 | nan | -1.57e-03 | 0.309 | 0 | 0 |
| prune | Ashton3 | qv_power_s2_6_g2_sg100 | qv_power_s2_6_g2 | yes | 100 | no | 0.176 | 0.152 | nan | -1.64e-03 | 0.871 | 3.55e-15 | 0 |
| prune | Ashton3 | qv_power_s2_6_g3_sg100 | qv_power_s2_6_g3 | yes | 100 | no | 0.174 | 0.179 | nan | -1.71e-03 | 0.221 | 0 | 0 |
| prune | Chong | qv_power_s2_6_g0p5_sg100 | qv_power_s2_6_g0p5 | yes | 100 | no | 0.324 | 0.237 | nan | 6.31e-03 | 1.156 | 0 | 0 |
| prune | Chong | qv_power_s2_6_g1_sg100 | qv_power_s2_6_g1 | yes | 100 | no | 0.293 | 0.134 | nan | 5.97e-03 | 0.727 | 0 | 0 |
| prune | Chong | qv_power_s2_6_g2_sg100 | qv_power_s2_6_g2 | yes | 100 | no | 0.263 | 0.071 | -9.45e-17 | 6.45e-03 | 1.024 | 3.55e-15 | 0 |
| prune | Chong | qv_power_s2_6_g3_sg100 | qv_power_s2_6_g3 | yes | 100 | no | 0.298 | -0.026 | nan | 6.11e-03 | 0.952 | 0 | 0 |
| prune | Igor | qv_power_s2_6_g0p5_sg100 | qv_power_s2_6_g0p5 | yes | 100 | no | 0.335 | 0.191 | nan | 6.25e-03 | 0.671 | 0 | 0 |
| prune | Igor | qv_power_s2_6_g1_sg100 | qv_power_s2_6_g1 | yes | 100 | no | 0.307 | 0.212 | nan | 3.99e-03 | 0.158 | 0 | 0 |
| prune | Igor | qv_power_s2_6_g2_sg100 | qv_power_s2_6_g2 | yes | 100 | no | 0.3 | 0.217 | nan | 1.10e-03 | 0.384 | 0 | 0 |
| prune | Igor | qv_power_s2_6_g3_sg100 | qv_power_s2_6_g3 | yes | 100 | yes | 0.344 | 0.223 | nan | -8.97e-04 | 0.068 | 0 | 0 |
| prune | John | qv_power_s2_6_g0p5_sg100 | qv_power_s2_6_g0p5 | yes | 100 | no | 0.239 | 0.258 | nan | -4.14e-04 | 0.239 | 7.11e-15 | 0 |
| prune | John | qv_power_s2_6_g1_sg100 | qv_power_s2_6_g1 | yes | 100 | yes | 0.118 | 0.227 | nan | -7.01e-05 | 0.054 | 7.11e-15 | 0 |
| prune | John | qv_power_s2_6_g2_sg100 | qv_power_s2_6_g2 | yes | 100 | yes | 0.45 | 0.352 | nan | -7.01e-05 | -0.04 | -7.11e-15 | 0 |
| prune | John | qv_power_s2_6_g3_sg100 | qv_power_s2_6_g3 | yes | 100 | yes | 0.191 | 0.277 | nan | -1.32e-06 | 0.04 | 7.11e-15 | 0 |
| prune | Kathy | qv_power_s2_6_g0p5_sg100 | qv_power_s2_6_g0p5 | yes | 100 | no | 0.357 | 0.325 | 4.07e-19 | 5.24e-03 | 0.517 | -7.11e-15 | 0 |
| prune | Kathy | qv_power_s2_6_g1_sg100 | qv_power_s2_6_g1 | yes | 100 | no | 0.263 | 0.368 | 6.30e-20 | 5.10e-03 | 0.465 | -7.11e-15 | 0 |
| prune | Kathy | qv_power_s2_6_g2_sg100 | qv_power_s2_6_g2 | yes | 100 | no | 0.312 | 0.348 | -1.23e-16 | 5.17e-03 | 0.294 | -7.11e-15 | 0 |
| prune | Kathy | qv_power_s2_6_g3_sg100 | qv_power_s2_6_g3 | yes | 100 | no | 0.288 | 0.422 | -1.23e-16 | 4.14e-03 | 0.197 | -7.11e-15 | 0 |
| prune | Pavel | qv_power_s2_6_g0p5_sg100 | qv_power_s2_6_g0p5 | yes | 100 | yes | 0.359 | 0.491 | 3.92e-17 | -7.84e-04 | -0.325 | 0 | 0 |
| prune | Pavel | qv_power_s2_6_g1_sg100 | qv_power_s2_6_g1 | yes | 100 | yes | 0.321 | 0.358 | 3.66e-17 | -2.34e-04 | -0.27 | 0 | 0 |
| prune | Pavel | qv_power_s2_6_g2_sg100 | qv_power_s2_6_g2 | yes | 100 | yes | 0.343 | 0.367 | -2.74e-17 | -5.09e-04 | -0.416 | 0 | 0 |
| prune | Pavel | qv_power_s2_6_g3_sg100 | qv_power_s2_6_g3 | yes | 100 | yes | 0.297 | 0.409 | -6.10e-17 | -3.88e-03 | -0.28 | 2.84e-14 | 0 |
| prune | Zohre | qv_power_s2_6_g0p5_sg100 | qv_power_s2_6_g0p5 | yes | 100 | yes | 0.294 | 0.204 | 7.61e-17 | 4.13e-03 | -1.672 | 0 | 0 |
| prune | Zohre | qv_power_s2_6_g1_sg100 | qv_power_s2_6_g1 | yes | 100 | yes | 0.345 | 0.216 | -4.34e-17 | 2.14e-03 | -1.595 | 0 | 0 |
| prune | Zohre | qv_power_s2_6_g2_sg100 | qv_power_s2_6_g2 | yes | 100 | yes | 0.285 | 0.189 | 4.52e-18 | 3.58e-03 | -2.012 | 3.55e-15 | 0 |
| prune | Zohre | qv_power_s2_6_g3_sg100 | qv_power_s2_6_g3 | yes | 100 | yes | 0.355 | 0.217 | -2.18e-17 | -6.14e-04 | -2.06 | 1.42e-14 | 0 |

## Leave-One-Subject-Out

| holdout | selected_config | pass_hard_gates | raw_jump_reduction | raw_speed_p999_reduction | delta_prec_x | delta_j30 |
| --- | --- | --- | --- | --- | --- | --- |
| Ashton3 | qv_power_s2_6_g3_sg100 | no | 0.174 | 0.179 | 0.221 | 0 |
| Chong | qv_power_s2_6_g3_sg100 | no | 0.298 | -0.026 | 0.952 | 0 |
| Igor | qv_power_s2_6_g3_sg100 | yes | 0.344 | 0.223 | 0.068 | 0 |
| John | qv_power_s2_6_g3_sg100 | yes | 0.191 | 0.277 | 0.04 | 7.11e-15 |
| Kathy | qv_power_s2_6_g3_sg100 | no | 0.288 | 0.422 | 0.197 | -7.11e-15 |
| Pavel | qv_power_s2_6_g3_sg100 | yes | 0.297 | 0.409 | -0.28 | 2.84e-14 |
| Zohre | qv_power_s2_6_g3_sg100 | yes | 0.355 | 0.217 | -2.06 | 1.42e-14 |
