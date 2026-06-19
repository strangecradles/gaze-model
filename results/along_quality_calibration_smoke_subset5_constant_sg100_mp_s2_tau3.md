# Along-Quality Calibration

Generated: 2026-06-17T23:37:10

Hard gates per subject: r_dot_x >= baseline - 0.02; valid_frac >= baseline - 0.03; prec_x <= baseline + 0.10; j30 <= baseline + 0.05.

Primary improvement targets: median raw >=3 px jump fraction and median p99.9 raw line-step speed should each fall by at least 10%.

## Config Summary

| stage | variant_id | config_id | slew_gate | slew_max_deg_s | hypothesis_velocity_cost | hypothesis_acceleration_cost | motion_prior | motion_prior_sigma_rows | motion_prior_tau_ms | motion_prior_ncc_thr | n_subjects | pass_all | median_score | median_raw_jump_reduction | median_raw_speed_p999_reduction | median_delta_prec_x | median_delta_j30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prune | constant_sg100_mp_s2_tau3 | constant | yes | 100 | 0 | 0 | yes | 2 | 3 | 0.2 | 2 | yes | 0.348 | 0.261 | 0.072 | -0.165 | -0.085 |

## Subject Rows

| stage | subject | variant_id | config_id | slew_gate | slew_max_deg_s | hypothesis_velocity_cost | hypothesis_acceleration_cost | motion_prior | motion_prior_sigma_rows | motion_prior_tau_ms | motion_prior_ncc_thr | pass_hard_gates | raw_jump_reduction | raw_speed_p999_reduction | delta_r_dot_x | delta_valid_frac | delta_prec_x | delta_j30 | oculo_j30_reduction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prune | Ashton3 | constant_sg100_mp_s2_tau3 | constant | yes | 100 | 0 | 0 | yes | 2 | 3 | 0.2 | yes | 0.24 | 0.028 | -3.58e-03 | -3.53e-03 | -0.125 | -0.021 | -0.013 |
| prune | Chong | constant_sg100_mp_s2_tau3 | constant | yes | 100 | 0 | 0 | yes | 2 | 3 | 0.2 | yes | 0.282 | 0.116 | -1.56e-03 | -6.44e-03 | -0.204 | -0.149 | 0.129 |
