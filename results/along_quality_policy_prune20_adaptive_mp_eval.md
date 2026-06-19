# Along-Quality Calibration

Generated: 2026-06-18T00:43:56

Hard gates per subject: r_dot_x >= baseline - 0.02; valid_frac >= baseline - 0.03; prec_x <= baseline + 0.10; j30 <= baseline + 0.05.

Primary improvement targets: median raw >=3 px jump fraction and median p99.9 raw line-step speed should each fall by at least 10%.

## Config Summary

| stage | variant_id | config_id | slew_gate | slew_max_deg_s | hypothesis_velocity_cost | hypothesis_acceleration_cost | motion_prior | motion_prior_sigma_rows | motion_prior_tau_ms | motion_prior_ncc_thr | n_subjects | pass_all | median_score | median_raw_jump_reduction | median_raw_speed_p999_reduction | median_delta_prec_x | median_delta_j30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prune | constant_sg100_mp_s4_tau6 | constant | yes | 100 | 0 | 0 | yes | 4 | 6 | 0.2 | 1 | yes | 0.534 | 0.391 | 0.137 | -0.166 | -0.063 |
| prune | constant_sg100_mp_s2_tau6 | constant | yes | 100 | 0 | 0 | yes | 2 | 6 | 0.2 | 3 | yes | 0.472 | 0.295 | 0.197 | 0.018 | 7.68e-03 |
| prune | constant_sg100_mp_s2_tau3 | constant | yes | 100 | 0 | 0 | yes | 2 | 3 | 0.2 | 3 | yes | 0.448 | 0.292 | 0.155 | -0.062 | -0.037 |

## Subject Rows

| stage | subject | variant_id | config_id | slew_gate | slew_max_deg_s | hypothesis_velocity_cost | hypothesis_acceleration_cost | motion_prior | motion_prior_sigma_rows | motion_prior_tau_ms | motion_prior_ncc_thr | pass_hard_gates | raw_jump_reduction | raw_speed_p999_reduction | delta_r_dot_x | delta_valid_frac | delta_prec_x | delta_j30 | oculo_j30_reduction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prune | Ashton3 | constant_sg100_mp_s2_tau3 | constant | yes | 100 | 0 | 0 | yes | 2 | 3 | 0.2 | yes | 0.289 | 0.128 | 3.31e-04 | -3.50e-03 | -0.062 | -0.037 | 0.02 |
| prune | Chong | constant_sg100_mp_s2_tau6 | constant | yes | 100 | 0 | 0 | yes | 2 | 6 | 0.2 | yes | 0.286 | 0.197 | 3.00e-04 | -6.51e-03 | 0.018 | 7.68e-03 | -0.047 |
| prune | Igor | constant_sg100_mp_s2_tau3 | constant | yes | 100 | 0 | 0 | yes | 2 | 3 | 0.2 | yes | 0.292 | 0.155 | -2.44e-03 | -0.011 | -0.019 | -0.014 | 5.41e-03 |
| prune | John | constant_sg100_mp_s2_tau3 | constant | yes | 100 | 0 | 0 | yes | 2 | 3 | 0.2 | yes | 0.303 | 0.175 | 3.21e-04 | -3.91e-03 | -0.078 | -0.044 | 0.014 |
| prune | Kathy | constant_sg100_mp_s2_tau6 | constant | yes | 100 | 0 | 0 | yes | 2 | 6 | 0.2 | yes | 0.295 | 0.205 | 4.50e-04 | -6.90e-03 | -0.045 | 0.021 | -0.053 |
| prune | Pavel | constant_sg100_mp_s2_tau6 | constant | yes | 100 | 0 | 0 | yes | 2 | 6 | 0.2 | yes | 0.31 | 0.157 | 2.29e-04 | -0.013 | 0.081 | 1.55e-03 | -0.096 |
| prune | Zohre | constant_sg100_mp_s4_tau6 | constant | yes | 100 | 0 | 0 | yes | 4 | 6 | 0.2 | yes | 0.391 | 0.137 | 4.51e-03 | -0.017 | -0.166 | -0.063 | 0.025 |
