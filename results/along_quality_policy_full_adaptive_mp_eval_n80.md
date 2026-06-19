# Along-Quality Calibration

Generated: 2026-06-18T09:42:06

Hard gates per subject: r_dot_x >= baseline - 0.02; valid_frac >= baseline - 0.03; prec_x <= baseline + 0.10; j30 <= baseline + 0.05.

Primary improvement targets: median raw >=3 px jump fraction and median p99.9 raw line-step speed should each fall by at least 10%.

## Config Summary

| stage | variant_id | config_id | slew_gate | slew_max_deg_s | hypothesis_velocity_cost | hypothesis_acceleration_cost | motion_prior | motion_prior_sigma_rows | motion_prior_tau_ms | motion_prior_ncc_thr | n_subjects | pass_all | median_score | median_raw_jump_reduction | median_raw_speed_p999_reduction | median_delta_prec_x | median_delta_j30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | constant_sg100_mp_s4_tau6 | constant | yes | 100 | 0 | 0 | yes | 4 | 6 | 0.2 | 1 | yes | 0.52 | 0.406 | 0.126 | -0.023 | -0.025 |
| full | constant_sg100_mp_s2_tau6 | constant | yes | 100 | 0 | 0 | yes | 2 | 6 | 0.2 | 3 | no | 0.461 | 0.313 | 0.174 | 0.098 | 0.036 |
| full | constant_sg100_mp_s2_tau3 | constant | yes | 100 | 0 | 0 | yes | 2 | 3 | 0.2 | 3 | no | 0.446 | 0.336 | 0.147 | 0.052 | 0.017 |

## Subject Rows

| stage | subject | variant_id | config_id | slew_gate | slew_max_deg_s | hypothesis_velocity_cost | hypothesis_acceleration_cost | motion_prior | motion_prior_sigma_rows | motion_prior_tau_ms | motion_prior_ncc_thr | pass_hard_gates | raw_jump_reduction | raw_speed_p999_reduction | delta_r_dot_x | delta_valid_frac | delta_prec_x | delta_j30 | oculo_j30_reduction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | Ashton3 | constant_sg100_mp_s2_tau3 | constant | yes | 100 | 0 | 0 | yes | 2 | 3 | 0.2 | yes | 0.361 | 0.159 | 2.43e-04 | -9.97e-03 | 0.052 | 0.017 | -0.021 |
| full | Chong | constant_sg100_mp_s2_tau6 | constant | yes | 100 | 0 | 0 | yes | 2 | 6 | 0.2 | yes | 0.352 | 0.174 | -1.65e-04 | -0.014 | 0.098 | -0.034 | 0.02 |
| full | Igor | constant_sg100_mp_s2_tau3 | constant | yes | 100 | 0 | 0 | yes | 2 | 3 | 0.2 | no | 0.336 | 0.141 | -6.46e-04 | -0.012 | 0.125 | 0.065 | -0.089 |
| full | John | constant_sg100_mp_s2_tau3 | constant | yes | 100 | 0 | 0 | yes | 2 | 3 | 0.2 | yes | 0.306 | 0.147 | 1.49e-04 | -4.29e-03 | -0.034 | -0.021 | -0.026 |
| full | Kathy | constant_sg100_mp_s2_tau6 | constant | yes | 100 | 0 | 0 | yes | 2 | 6 | 0.2 | yes | 0.284 | 0.19 | 1.42e-04 | -6.98e-03 | 0.022 | 0.036 | -0.049 |
| full | Pavel | constant_sg100_mp_s2_tau6 | constant | yes | 100 | 0 | 0 | yes | 2 | 6 | 0.2 | no | 0.313 | 0.172 | 2.46e-04 | -0.012 | 0.128 | 0.098 | -0.076 |
| full | Zohre | constant_sg100_mp_s4_tau6 | constant | yes | 100 | 0 | 0 | yes | 4 | 6 | 0.2 | yes | 0.406 | 0.126 | 2.20e-03 | -0.018 | -0.023 | -0.025 | -0.047 |
