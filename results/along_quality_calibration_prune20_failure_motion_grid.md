# Along-Quality Calibration

Generated: 2026-06-18T00:29:32

Hard gates per subject: r_dot_x >= baseline - 0.02; valid_frac >= baseline - 0.03; prec_x <= baseline + 0.10; j30 <= baseline + 0.05.

Primary improvement targets: median raw >=3 px jump fraction and median p99.9 raw line-step speed should each fall by at least 10%.

## Config Summary

| stage | variant_id | config_id | slew_gate | slew_max_deg_s | hypothesis_velocity_cost | hypothesis_acceleration_cost | motion_prior | motion_prior_sigma_rows | motion_prior_tau_ms | motion_prior_ncc_thr | n_subjects | pass_all | median_score | median_raw_jump_reduction | median_raw_speed_p999_reduction | median_delta_prec_x | median_delta_j30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prune | constant_sg100_mp_s2_tau6 | constant | yes | 100 | 0 | 0 | yes | 2 | 6 | 0.2 | 3 | no | 0.443 | 0.31 | 0.157 | -0.045 | 0.021 |
| prune | constant_sg100_mp_s3_tau3 | constant | yes | 100 | 0 | 0 | yes | 3 | 3 | 0.2 | 3 | no | -9.475 | 0.306 | 0.174 | -0.045 | 0.152 |
| prune | constant_sg100_mp_s1p5_tau3 | constant | yes | 100 | 0 | 0 | yes | 1.5 | 3 | 0.2 | 3 | no | -9.513 | 0.301 | 0.168 | 0.047 | 0.081 |
| prune | constant_sg100_mp_s4_tau6 | constant | yes | 100 | 0 | 0 | yes | 4 | 6 | 0.2 | 3 | no | -9.526 | 0.306 | 0.148 | -0.039 | 0.09 |
| prune | constant_sg100_mp_s2_tau3 | constant | yes | 100 | 0 | 0 | yes | 2 | 3 | 0.2 | 3 | no | -9.543 | 0.308 | 0.16 | -0.071 | 0.08 |
| prune | constant_sg100_mp_s3_tau6 | constant | yes | 100 | 0 | 0 | yes | 3 | 6 | 0.2 | 3 | no | -9.565 | 0.308 | 0.138 | -0.052 | 0.131 |

## Subject Rows

| stage | subject | variant_id | config_id | slew_gate | slew_max_deg_s | hypothesis_velocity_cost | hypothesis_acceleration_cost | motion_prior | motion_prior_sigma_rows | motion_prior_tau_ms | motion_prior_ncc_thr | pass_hard_gates | raw_jump_reduction | raw_speed_p999_reduction | delta_r_dot_x | delta_valid_frac | delta_prec_x | delta_j30 | oculo_j30_reduction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prune | Kathy | constant_sg100_mp_s1p5_tau3 | constant | yes | 100 | 0 | 0 | yes | 1.5 | 3 | 0.2 | no | 0.299 | 0.181 | 4.96e-04 | -6.96e-03 | 0.047 | 0.114 | 0.032 |
| prune | Kathy | constant_sg100_mp_s2_tau3 | constant | yes | 100 | 0 | 0 | yes | 2 | 3 | 0.2 | no | 0.289 | 0.192 | 2.40e-04 | -6.92e-03 | -0.071 | 0.08 | -0.095 |
| prune | Kathy | constant_sg100_mp_s2_tau6 | constant | yes | 100 | 0 | 0 | yes | 2 | 6 | 0.2 | yes | 0.295 | 0.205 | 4.50e-04 | -6.90e-03 | -0.045 | 0.021 | -0.053 |
| prune | Kathy | constant_sg100_mp_s3_tau3 | constant | yes | 100 | 0 | 0 | yes | 3 | 3 | 0.2 | yes | 0.292 | 0.189 | 2.76e-04 | -6.65e-03 | -0.045 | 0.016 | 0.016 |
| prune | Kathy | constant_sg100_mp_s3_tau6 | constant | yes | 100 | 0 | 0 | yes | 3 | 6 | 0.2 | no | 0.307 | 0.138 | 4.84e-04 | -7.11e-03 | -0.052 | 0.131 | -0.061 |
| prune | Kathy | constant_sg100_mp_s4_tau6 | constant | yes | 100 | 0 | 0 | yes | 4 | 6 | 0.2 | no | 0.298 | 0.184 | 2.95e-04 | -6.75e-03 | -0.039 | 0.09 | -0.034 |
| prune | Pavel | constant_sg100_mp_s1p5_tau3 | constant | yes | 100 | 0 | 0 | yes | 1.5 | 3 | 0.2 | no | 0.301 | 0.168 | 8.02e-04 | -0.013 | 0.118 | 0.057 | -0.126 |
| prune | Pavel | constant_sg100_mp_s2_tau3 | constant | yes | 100 | 0 | 0 | yes | 2 | 3 | 0.2 | no | 0.308 | 0.16 | 1.26e-04 | -0.013 | 0.13 | 0.024 | -0.173 |
| prune | Pavel | constant_sg100_mp_s2_tau6 | constant | yes | 100 | 0 | 0 | yes | 2 | 6 | 0.2 | yes | 0.31 | 0.157 | 2.29e-04 | -0.013 | 0.081 | 1.55e-03 | -0.096 |
| prune | Pavel | constant_sg100_mp_s3_tau3 | constant | yes | 100 | 0 | 0 | yes | 3 | 3 | 0.2 | no | 0.306 | 0.174 | 1.59e-04 | -0.013 | 0.098 | 0.152 | -0.126 |
| prune | Pavel | constant_sg100_mp_s3_tau6 | constant | yes | 100 | 0 | 0 | yes | 3 | 6 | 0.2 | no | 0.308 | 0.168 | 6.40e-04 | -0.013 | 0.115 | 0.152 | -0.163 |
| prune | Pavel | constant_sg100_mp_s4_tau6 | constant | yes | 100 | 0 | 0 | yes | 4 | 6 | 0.2 | no | 0.306 | 0.148 | 6.87e-04 | -0.013 | -0.011 | 0.138 | -0.16 |
| prune | Zohre | constant_sg100_mp_s1p5_tau3 | constant | yes | 100 | 0 | 0 | yes | 1.5 | 3 | 0.2 | no | 0.391 | 0.136 | 5.12e-03 | -0.018 | -0.302 | 0.081 | 0.035 |
| prune | Zohre | constant_sg100_mp_s2_tau3 | constant | yes | 100 | 0 | 0 | yes | 2 | 3 | 0.2 | no | 0.391 | 0.128 | -5.77e-03 | -0.018 | -0.223 | 0.191 | 0.095 |
| prune | Zohre | constant_sg100_mp_s2_tau6 | constant | yes | 100 | 0 | 0 | yes | 2 | 6 | 0.2 | no | 0.389 | 0.142 | 7.44e-03 | -0.018 | -0.187 | 0.095 | 0.017 |
| prune | Zohre | constant_sg100_mp_s3_tau3 | constant | yes | 100 | 0 | 0 | yes | 3 | 3 | 0.2 | no | 0.39 | 0.13 | 4.18e-03 | -0.017 | -0.093 | 0.242 | 0.018 |
| prune | Zohre | constant_sg100_mp_s3_tau6 | constant | yes | 100 | 0 | 0 | yes | 3 | 6 | 0.2 | no | 0.394 | 0.122 | 6.35e-03 | -0.017 | -0.227 | 0.055 | 0.024 |
| prune | Zohre | constant_sg100_mp_s4_tau6 | constant | yes | 100 | 0 | 0 | yes | 4 | 6 | 0.2 | yes | 0.391 | 0.137 | 4.51e-03 | -0.017 | -0.166 | -0.063 | 0.025 |

## Leave-One-Subject-Out

| holdout | selected_config | pass_hard_gates | raw_jump_reduction | raw_speed_p999_reduction | delta_prec_x | delta_j30 |
| --- | --- | --- | --- | --- | --- | --- |
| Kathy | constant_sg100_mp_s2_tau6 | yes | 0.295 | 0.205 | -0.045 | 0.021 |
| Pavel | constant_sg100_mp_s2_tau6 | yes | 0.31 | 0.157 | 0.081 | 1.55e-03 |
| Zohre | constant_sg100_mp_s2_tau6 | no | 0.389 | 0.142 | -0.187 | 0.095 |
