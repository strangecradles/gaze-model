# Along-Quality Calibration

Generated: 2026-06-18T11:44:09

Hard gates per subject: r_dot_x >= baseline - 0.02; valid_frac >= baseline - 0.03; prec_x <= baseline + 0.10; j30 <= baseline + 0.05.

Primary improvement targets: median raw >=3 px jump fraction and median p99.9 raw line-step speed should each fall by at least 10%.

## Config Summary

| stage | variant_id | config_id | slew_gate | slew_max_deg_s | lag_ms | hypothesis_transition_sigma_rows | hypothesis_obs_weight | hypothesis_velocity_cost | hypothesis_acceleration_cost | motion_prior | motion_prior_sigma_rows | motion_prior_tau_ms | motion_prior_ncc_thr | n_subjects | pass_all | median_score | median_raw_jump_reduction | median_raw_speed_p999_reduction | median_delta_prec_x | median_delta_j30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | constant_ow6_sg75_mp_s2_tau6 | constant | yes | 75 | 1 | 3 | 6 | 0 | 0 | yes | 2 | 6 | 0.2 | 1 | no | -9.544 | 0.303 | 0.163 | 0.128 | 0.055 |
| full | constant_ow8_sg75_mp_s2_tau6 | constant | yes | 75 | 1 | 3 | 8 | 0 | 0 | yes | 2 | 6 | 0.2 | 1 | no | -9.546 | 0.301 | 0.163 | 0.127 | 0.066 |
| full | constant_ow12_sg75_mp_s2_tau6 | constant | yes | 75 | 1 | 3 | 12 | 0 | 0 | yes | 2 | 6 | 0.2 | 1 | no | -9.553 | 0.3 | 0.162 | 0.128 | 0.054 |

## Subject Rows

| stage | subject | variant_id | config_id | slew_gate | slew_max_deg_s | lag_ms | hypothesis_transition_sigma_rows | hypothesis_obs_weight | hypothesis_velocity_cost | hypothesis_acceleration_cost | motion_prior | motion_prior_sigma_rows | motion_prior_tau_ms | motion_prior_ncc_thr | pass_hard_gates | raw_jump_reduction | raw_speed_p999_reduction | delta_r_dot_x | delta_valid_frac | delta_prec_x | delta_j30 | oculo_j30_reduction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | Pavel | constant_ow6_sg75_mp_s2_tau6 | constant | yes | 75 | 1 | 3 | 6 | 0 | 0 | yes | 2 | 6 | 0.2 | no | 0.303 | 0.163 | 2.61e-04 | -0.012 | 0.128 | 0.055 | -0.038 |
| full | Pavel | constant_ow8_sg75_mp_s2_tau6 | constant | yes | 75 | 1 | 3 | 8 | 0 | 0 | yes | 2 | 6 | 0.2 | no | 0.301 | 0.163 | 2.67e-04 | -0.012 | 0.127 | 0.066 | -0.038 |
| full | Pavel | constant_ow12_sg75_mp_s2_tau6 | constant | yes | 75 | 1 | 3 | 12 | 0 | 0 | yes | 2 | 6 | 0.2 | no | 0.3 | 0.162 | 2.72e-04 | -0.012 | 0.128 | 0.054 | -0.059 |
