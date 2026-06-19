# Along-Quality Calibration

Generated: 2026-06-18T09:56:00

Hard gates per subject: r_dot_x >= baseline - 0.02; valid_frac >= baseline - 0.03; prec_x <= baseline + 0.10; j30 <= baseline + 0.05.

Primary improvement targets: median raw >=3 px jump fraction and median p99.9 raw line-step speed should each fall by at least 10%.

## Config Summary

| stage | variant_id | config_id | slew_gate | slew_max_deg_s | hypothesis_velocity_cost | hypothesis_acceleration_cost | motion_prior | motion_prior_sigma_rows | motion_prior_tau_ms | motion_prior_ncc_thr | n_subjects | pass_all | median_score | median_raw_jump_reduction | median_raw_speed_p999_reduction | median_delta_prec_x | median_delta_j30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | constant_sg100 | constant | yes | 100 | 0 | 0 | no |  |  |  | 2 | no | -9.544 | 0.327 | 0.153 | 0.114 | 0.105 |

## Subject Rows

| stage | subject | variant_id | config_id | slew_gate | slew_max_deg_s | hypothesis_velocity_cost | hypothesis_acceleration_cost | motion_prior | motion_prior_sigma_rows | motion_prior_tau_ms | motion_prior_ncc_thr | pass_hard_gates | raw_jump_reduction | raw_speed_p999_reduction | delta_r_dot_x | delta_valid_frac | delta_prec_x | delta_j30 | oculo_j30_reduction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | Igor | constant_sg100 | constant | yes | 100 | 0 | 0 | no |  |  |  | no | 0.337 | 0.15 | 7.32e-03 | -0.012 | 0.089 | 0.077 | -0.101 |
| full | Pavel | constant_sg100 | constant | yes | 100 | 0 | 0 | no |  |  |  | no | 0.317 | 0.157 | -8.22e-05 | -0.012 | 0.138 | 0.134 | -0.087 |
