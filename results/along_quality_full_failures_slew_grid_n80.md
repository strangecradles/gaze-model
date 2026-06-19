# Along-Quality Calibration

Generated: 2026-06-18T10:49:22

Hard gates per subject: r_dot_x >= baseline - 0.02; valid_frac >= baseline - 0.03; prec_x <= baseline + 0.10; j30 <= baseline + 0.05.

Primary improvement targets: median raw >=3 px jump fraction and median p99.9 raw line-step speed should each fall by at least 10%.

## Config Summary

| stage | variant_id | config_id | slew_gate | slew_max_deg_s | hypothesis_velocity_cost | hypothesis_acceleration_cost | motion_prior | motion_prior_sigma_rows | motion_prior_tau_ms | motion_prior_ncc_thr | n_subjects | pass_all | median_score | median_raw_jump_reduction | median_raw_speed_p999_reduction | median_delta_prec_x | median_delta_j30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | constant_sg125 | constant | yes | 125 | 0 | 0 | no |  |  |  | 2 | no | -4.531 | 0.334 | 0.158 | 0.111 | 0.092 |
| full | constant_sg150 | constant | yes | 150 | 0 | 0 | no |  |  |  | 2 | no | -9.521 | 0.341 | 0.159 | 0.106 | 0.105 |
| full | constant_sg75 | constant | yes | 75 | 0 | 0 | no |  |  |  | 2 | no | -9.56 | 0.319 | 0.148 | 0.124 | 0.098 |
| full | constant_sg50 | constant | yes | 50 | 0 | 0 | no |  |  |  | 2 | no | -9.573 | 0.309 | 0.149 | 0.135 | 0.086 |

## Subject Rows

| stage | subject | variant_id | config_id | slew_gate | slew_max_deg_s | hypothesis_velocity_cost | hypothesis_acceleration_cost | motion_prior | motion_prior_sigma_rows | motion_prior_tau_ms | motion_prior_ncc_thr | pass_hard_gates | raw_jump_reduction | raw_speed_p999_reduction | delta_r_dot_x | delta_valid_frac | delta_prec_x | delta_j30 | oculo_j30_reduction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | Igor | constant_sg50 | constant | yes | 50 | 0 | 0 | no |  |  |  | no | 0.319 | 0.144 | 7.32e-03 | -0.012 | 0.097 | 0.05 | -0.115 |
| full | Igor | constant_sg75 | constant | yes | 75 | 0 | 0 | no |  |  |  | no | 0.329 | 0.142 | 7.31e-03 | -0.012 | 0.098 | 0.06 | -0.106 |
| full | Igor | constant_sg125 | constant | yes | 125 | 0 | 0 | no |  |  |  | yes | 0.345 | 0.152 | 7.27e-03 | -0.012 | 0.089 | 0.042 | -0.089 |
| full | Igor | constant_sg150 | constant | yes | 150 | 0 | 0 | no |  |  |  | no | 0.351 | 0.152 | 7.31e-03 | -0.012 | 0.083 | 0.066 | -0.071 |
| full | Pavel | constant_sg50 | constant | yes | 50 | 0 | 0 | no |  |  |  | no | 0.299 | 0.153 | 1.14e-04 | -0.012 | 0.173 | 0.123 | -0.132 |
| full | Pavel | constant_sg75 | constant | yes | 75 | 0 | 0 | no |  |  |  | no | 0.309 | 0.155 | -1.50e-04 | -0.012 | 0.149 | 0.136 | -0.109 |
| full | Pavel | constant_sg125 | constant | yes | 125 | 0 | 0 | no |  |  |  | no | 0.324 | 0.164 | -8.89e-05 | -0.012 | 0.132 | 0.141 | -0.095 |
| full | Pavel | constant_sg150 | constant | yes | 150 | 0 | 0 | no |  |  |  | no | 0.33 | 0.166 | -6.43e-05 | -0.012 | 0.129 | 0.145 | -0.095 |

## Leave-One-Subject-Out

| holdout | selected_config | pass_hard_gates | raw_jump_reduction | raw_speed_p999_reduction | delta_prec_x | delta_j30 |
| --- | --- | --- | --- | --- | --- | --- |
| Igor | constant_sg150 | no | 0.351 | 0.152 | 0.083 | 0.066 |
| Pavel | constant_sg125 | no | 0.324 | 0.164 | 0.132 | 0.141 |
