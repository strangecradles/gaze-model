# Along-Quality Calibration

Generated: 2026-06-18T12:25:55

Hard gates per subject: r_dot_x >= baseline - 0.02; valid_frac >= baseline - 0.03; prec_x <= baseline + 0.10; j30 <= baseline + 0.05.

Primary improvement targets: median raw >=3 px jump fraction and median p99.9 raw line-step speed should each fall by at least 10%.

## Config Summary

| stage | variant_id | config_id | slew_gate | slew_max_deg_s | lag_ms | hypothesis_transition_sigma_rows | hypothesis_obs_weight | hypothesis_blend_immediate | hypothesis_blend_delta_rows | hypothesis_blend_alpha | hypothesis_velocity_cost | hypothesis_acceleration_cost | motion_prior | motion_prior_sigma_rows | motion_prior_tau_ms | motion_prior_ncc_thr | n_subjects | pass_all | median_score | median_raw_jump_reduction | median_raw_speed_p999_reduction | median_raw_jump_return_reduction | median_delta_prec_x | median_delta_j30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | constant_sg100_mp_s4_tau6 | constant | yes | 100 | 1 | 3 | 4 | no |  |  | 0 | 0 | yes | 4 | 6 | 0.2 | 1 | yes | 0.52 | 0.406 | 0.126 | 0.696 | -0.023 | -0.025 |
| full | constant_sg100_mp_s2_tau6 | constant | yes | 100 | 1 | 3 | 4 | no |  |  | 0 | 0 | yes | 2 | 6 | 0.2 | 2 | yes | 0.496 | 0.318 | 0.182 | 0.628 | 0.06 | 8.59e-04 |
| full | constant_sg100_mp_s2_tau3 | constant | yes | 100 | 1 | 3 | 4 | no |  |  | 0 | 0 | yes | 2 | 3 | 0.2 | 2 | yes | 0.481 | 0.333 | 0.153 | 0.628 | 8.68e-03 | -2.28e-03 |
| full | constant_sg125 | constant | yes | 125 | 1 | 3 | 4 | no |  |  | 0 | 0 | no |  |  |  | 1 | yes | 0.475 | 0.345 | 0.152 | 0.661 | 0.089 | 0.042 |
| full | constant_bi_d12_a0p5_sg100_mp_s2_tau6 | constant | yes | 100 | 1 | 3 | 4 | yes | 12 | 0.5 | 0 | 0 | yes | 2 | 6 | 0.2 | 1 | yes | 0.376 | 0.159 | 0.24 | 0.351 | 0.045 | 0.04 |

## Subject Rows

| stage | subject | variant_id | config_id | slew_gate | slew_max_deg_s | lag_ms | hypothesis_transition_sigma_rows | hypothesis_obs_weight | hypothesis_blend_immediate | hypothesis_blend_delta_rows | hypothesis_blend_alpha | hypothesis_velocity_cost | hypothesis_acceleration_cost | motion_prior | motion_prior_sigma_rows | motion_prior_tau_ms | motion_prior_ncc_thr | pass_hard_gates | raw_jump_reduction | raw_speed_p999_reduction | raw_jump_return_reduction | delta_r_dot_x | delta_valid_frac | delta_prec_x | delta_j30 | oculo_j30_reduction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | Ashton3 | constant_sg100_mp_s2_tau3 | constant | yes | 100 | 1 | 3 | 4 | no |  |  | 0 | 0 | yes | 2 | 3 | 0.2 | yes | 0.361 | 0.159 | 0.654 | 2.43e-04 | -9.97e-03 | 0.052 | 0.017 | -0.021 |
| full | Chong | constant_sg100_mp_s2_tau6 | constant | yes | 100 | 1 | 3 | 4 | no |  |  | 0 | 0 | yes | 2 | 6 | 0.2 | yes | 0.352 | 0.174 | 0.659 | -1.65e-04 | -0.014 | 0.098 | -0.034 | 0.02 |
| full | Igor | constant_sg125 | constant | yes | 125 | 1 | 3 | 4 | no |  |  | 0 | 0 | no |  |  |  | yes | 0.345 | 0.152 | 0.661 | 7.27e-03 | -0.012 | 0.089 | 0.042 | -0.089 |
| full | John | constant_sg100_mp_s2_tau3 | constant | yes | 100 | 1 | 3 | 4 | no |  |  | 0 | 0 | yes | 2 | 3 | 0.2 | yes | 0.306 | 0.147 | 0.603 | 1.49e-04 | -4.29e-03 | -0.034 | -0.021 | -0.026 |
| full | Kathy | constant_sg100_mp_s2_tau6 | constant | yes | 100 | 1 | 3 | 4 | no |  |  | 0 | 0 | yes | 2 | 6 | 0.2 | yes | 0.284 | 0.19 | 0.597 | 1.42e-04 | -6.98e-03 | 0.022 | 0.036 | -0.049 |
| full | Pavel | constant_bi_d12_a0p5_sg100_mp_s2_tau6 | constant | yes | 100 | 1 | 3 | 4 | yes | 12 | 0.5 | 0 | 0 | yes | 2 | 6 | 0.2 | yes | 0.159 | 0.24 | 0.351 | 2.77e-04 | -0.012 | 0.045 | 0.04 | -0.094 |
| full | Zohre | constant_sg100_mp_s4_tau6 | constant | yes | 100 | 1 | 3 | 4 | no |  |  | 0 | 0 | yes | 4 | 6 | 0.2 | yes | 0.406 | 0.126 | 0.696 | 2.20e-03 | -0.018 | -0.023 | -0.025 | -0.047 |

## Leave-One-Subject-Out

| holdout | selected_config | pass_hard_gates | raw_jump_reduction | raw_speed_p999_reduction | delta_prec_x | delta_j30 |
| --- | --- | --- | --- | --- | --- | --- |
| Kathy | constant_sg100_mp_s2_tau6 | yes | 0.284 | 0.19 | 0.022 | 0.036 |
