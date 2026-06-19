# Along-Quality Calibration

Generated: 2026-06-17T23:58:10

Hard gates per subject: r_dot_x >= baseline - 0.02; valid_frac >= baseline - 0.03; prec_x <= baseline + 0.10; j30 <= baseline + 0.05.

Primary improvement targets: median raw >=3 px jump fraction and median p99.9 raw line-step speed should each fall by at least 10%.

## Config Summary

| stage | variant_id | config_id | slew_gate | slew_max_deg_s | hypothesis_velocity_cost | hypothesis_acceleration_cost | motion_prior | motion_prior_sigma_rows | motion_prior_tau_ms | motion_prior_ncc_thr | n_subjects | pass_all | median_score | median_raw_jump_reduction | median_raw_speed_p999_reduction | median_delta_prec_x | median_delta_j30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prune | constant_sg100_mp_s2_tau3 | constant | yes | 100 | 0 | 0 | yes | 2 | 3 | 0.2 | 7 | no | 0.422 | 0.3 | 0.16 | -0.071 | -5.46e-03 |

## Subject Rows

| stage | subject | variant_id | config_id | slew_gate | slew_max_deg_s | hypothesis_velocity_cost | hypothesis_acceleration_cost | motion_prior | motion_prior_sigma_rows | motion_prior_tau_ms | motion_prior_ncc_thr | pass_hard_gates | raw_jump_reduction | raw_speed_p999_reduction | delta_r_dot_x | delta_valid_frac | delta_prec_x | delta_j30 | oculo_j30_reduction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prune | Ashton3 | constant_sg100_mp_s2_tau3 | constant | yes | 100 | 0 | 0 | yes | 2 | 3 | 0.2 | yes | 0.289 | 0.128 | 3.31e-04 | -3.50e-03 | -0.062 | -0.037 | 0.02 |
| prune | Chong | constant_sg100_mp_s2_tau3 | constant | yes | 100 | 0 | 0 | yes | 2 | 3 | 0.2 | yes | 0.3 | 0.172 | 6.91e-04 | -6.27e-03 | -0.096 | -5.46e-03 | -0.034 |
| prune | Igor | constant_sg100_mp_s2_tau3 | constant | yes | 100 | 0 | 0 | yes | 2 | 3 | 0.2 | yes | 0.292 | 0.155 | -2.44e-03 | -0.011 | -0.019 | -0.014 | 5.41e-03 |
| prune | John | constant_sg100_mp_s2_tau3 | constant | yes | 100 | 0 | 0 | yes | 2 | 3 | 0.2 | yes | 0.303 | 0.175 | 3.21e-04 | -3.91e-03 | -0.078 | -0.044 | 0.014 |
| prune | Kathy | constant_sg100_mp_s2_tau3 | constant | yes | 100 | 0 | 0 | yes | 2 | 3 | 0.2 | no | 0.289 | 0.192 | 2.40e-04 | -6.92e-03 | -0.071 | 0.08 | -0.095 |
| prune | Pavel | constant_sg100_mp_s2_tau3 | constant | yes | 100 | 0 | 0 | yes | 2 | 3 | 0.2 | no | 0.308 | 0.16 | 1.26e-04 | -0.013 | 0.13 | 0.024 | -0.173 |
| prune | Zohre | constant_sg100_mp_s2_tau3 | constant | yes | 100 | 0 | 0 | yes | 2 | 3 | 0.2 | no | 0.391 | 0.128 | -5.77e-03 | -0.018 | -0.223 | 0.191 | 0.095 |
