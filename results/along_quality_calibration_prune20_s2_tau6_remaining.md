# Along-Quality Calibration

Generated: 2026-06-18T00:38:54

Hard gates per subject: r_dot_x >= baseline - 0.02; valid_frac >= baseline - 0.03; prec_x <= baseline + 0.10; j30 <= baseline + 0.05.

Primary improvement targets: median raw >=3 px jump fraction and median p99.9 raw line-step speed should each fall by at least 10%.

## Config Summary

| stage | variant_id | config_id | slew_gate | slew_max_deg_s | hypothesis_velocity_cost | hypothesis_acceleration_cost | motion_prior | motion_prior_sigma_rows | motion_prior_tau_ms | motion_prior_ncc_thr | n_subjects | pass_all | median_score | median_raw_jump_reduction | median_raw_speed_p999_reduction | median_delta_prec_x | median_delta_j30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prune | constant_sg100_mp_s2_tau6 | constant | yes | 100 | 0 | 0 | yes | 2 | 6 | 0.2 | 4 | no | -4.576 | 0.289 | 0.166 | -0.042 | 0.031 |

## Subject Rows

| stage | subject | variant_id | config_id | slew_gate | slew_max_deg_s | hypothesis_velocity_cost | hypothesis_acceleration_cost | motion_prior | motion_prior_sigma_rows | motion_prior_tau_ms | motion_prior_ncc_thr | pass_hard_gates | raw_jump_reduction | raw_speed_p999_reduction | delta_r_dot_x | delta_valid_frac | delta_prec_x | delta_j30 | oculo_j30_reduction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prune | Ashton3 | constant_sg100_mp_s2_tau6 | constant | yes | 100 | 0 | 0 | yes | 2 | 6 | 0.2 | no | 0.281 | 0.138 | -1.80e-05 | -3.52e-03 | -0.107 | 0.054 | -0.025 |
| prune | Chong | constant_sg100_mp_s2_tau6 | constant | yes | 100 | 0 | 0 | yes | 2 | 6 | 0.2 | yes | 0.286 | 0.197 | 3.00e-04 | -6.51e-03 | 0.018 | 7.68e-03 | -0.047 |
| prune | Igor | constant_sg100_mp_s2_tau6 | constant | yes | 100 | 0 | 0 | yes | 2 | 6 | 0.2 | no | 0.295 | 0.194 | -2.44e-03 | -0.011 | 8.92e-03 | 0.182 | -0.09 |
| prune | John | constant_sg100_mp_s2_tau6 | constant | yes | 100 | 0 | 0 | yes | 2 | 6 | 0.2 | yes | 0.291 | 0.109 | 3.36e-04 | -3.74e-03 | -0.093 | -5.93e-03 | -0.069 |
