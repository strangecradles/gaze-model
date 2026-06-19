# Along-Quality Calibration

Generated: 2026-06-18T11:19:57

Hard gates per subject: r_dot_x >= baseline - 0.02; valid_frac >= baseline - 0.03; prec_x <= baseline + 0.10; j30 <= baseline + 0.05.

Primary improvement targets: median raw >=3 px jump fraction and median p99.9 raw line-step speed should each fall by at least 10%.

## Config Summary

| stage | variant_id | config_id | slew_gate | slew_max_deg_s | hypothesis_velocity_cost | hypothesis_acceleration_cost | motion_prior | motion_prior_sigma_rows | motion_prior_tau_ms | motion_prior_ncc_thr | n_subjects | pass_all | median_score | median_raw_jump_reduction | median_raw_speed_p999_reduction | median_delta_prec_x | median_delta_j30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | constant_sg150_mp_s2_tau6 | constant | yes | 150 | 0 | 0 | yes | 2 | 6 | 0.2 | 1 | no | -9.508 | 0.328 | 0.179 | 0.115 | 0.089 |
| full | constant_sg125_mp_s2_tau6 | constant | yes | 125 | 0 | 0 | yes | 2 | 6 | 0.2 | 1 | no | -9.516 | 0.322 | 0.176 | 0.123 | 0.099 |
| full | constant_sg100_mp_s2_tau6 | constant | yes | 100 | 0 | 0 | yes | 2 | 6 | 0.2 | 1 | no | -9.534 | 0.313 | 0.172 | 0.128 | 0.098 |
| full | constant_sg75_mp_s2_tau6 | constant | yes | 75 | 0 | 0 | yes | 2 | 6 | 0.2 | 1 | no | -9.547 | 0.305 | 0.163 | 0.127 | 0.08 |
| full | constant_sg50_mp_s2_tau6 | constant | yes | 50 | 0 | 0 | yes | 2 | 6 | 0.2 | 1 | no | -9.575 | 0.295 | 0.16 | 0.128 | 0.099 |

## Subject Rows

| stage | subject | variant_id | config_id | slew_gate | slew_max_deg_s | hypothesis_velocity_cost | hypothesis_acceleration_cost | motion_prior | motion_prior_sigma_rows | motion_prior_tau_ms | motion_prior_ncc_thr | pass_hard_gates | raw_jump_reduction | raw_speed_p999_reduction | delta_r_dot_x | delta_valid_frac | delta_prec_x | delta_j30 | oculo_j30_reduction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | Pavel | constant_sg50_mp_s2_tau6 | constant | yes | 50 | 0 | 0 | yes | 2 | 6 | 0.2 | no | 0.295 | 0.16 | 2.58e-04 | -0.012 | 0.128 | 0.099 | -0.12 |
| full | Pavel | constant_sg75_mp_s2_tau6 | constant | yes | 75 | 0 | 0 | yes | 2 | 6 | 0.2 | no | 0.305 | 0.163 | 2.61e-04 | -0.012 | 0.127 | 0.08 | -0.061 |
| full | Pavel | constant_sg100_mp_s2_tau6 | constant | yes | 100 | 0 | 0 | yes | 2 | 6 | 0.2 | no | 0.313 | 0.172 | 2.46e-04 | -0.012 | 0.128 | 0.098 | -0.076 |
| full | Pavel | constant_sg125_mp_s2_tau6 | constant | yes | 125 | 0 | 0 | yes | 2 | 6 | 0.2 | no | 0.322 | 0.176 | 2.57e-04 | -0.012 | 0.123 | 0.099 | -0.051 |
| full | Pavel | constant_sg150_mp_s2_tau6 | constant | yes | 150 | 0 | 0 | yes | 2 | 6 | 0.2 | no | 0.328 | 0.179 | 2.66e-04 | -0.012 | 0.115 | 0.089 | -0.059 |
