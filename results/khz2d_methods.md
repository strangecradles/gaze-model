# kHz 2D Gaze From x-Scan Lines + 2D SLO — Method Comparison (Testbed A)

**Task**: reconstruct 2D gaze at kHz output rate from (i) the ~12 kHz fast-axis line stream and (ii) the slow-axis 2D SLO frames (14.63 Hz measured, nominal 20 Hz).

**Testbed A** = the `test1` pursuit raster: its columns ARE the 11,823 Hz x-scan (808 sweeps/frame), its frames ARE the 2D SLO. References: the 0.2 Hz pursuit dot (target, arcmin) and the ~32.5 Hz machine pupil tracker. Shared clock offset OFF = 2.50 s (estimated once at harness level, frozen for all methods; it absorbs the hardware sync offset + mean pursuit latency).

Evaluation protocol (identical for every method): 2 ms smoothing, 0.05 Hz drift removal, per-axis affine calibration to the dot on valid samples, then r + RMS vs dot, r vs tracker, precision = RMS of >25 ms detail. The dot is the TARGET, not the eye: r ~= 0.9 horizontal is the practical ceiling (pursuit lag, catch-up saccades).

## Method table

| method | rate (Hz) | r dot x | r dot y | r trk x | r trk y | RMS x (') | RMS y (') | prec x (') | prec y (') | valid |
|---|---|---|---|---|---|---|---|---|---|---|
| M0 SLO frames only (chain) | 15 | 0.906 | 0.739 | 0.513 | 0.303 | 16.0 | 19.2 | nan | nan | 72% |
| M1 strips S=20 | 585 | 0.877 | 0.743 | 0.650 | 0.327 | 19.9 | 19.0 | 4.29 | 1.22 | 71% |
| M1 strips S=8 | 1478 | 0.878 | 0.737 | 0.642 | 0.334 | 19.8 | 19.3 | 4.39 | 1.25 | 72% |
| M1 strips S=4 | 2956 | 0.875 | 0.737 | 0.643 | 0.334 | 20.0 | 19.3 | 4.52 | 1.27 | 73% |
| M1 strips S=2 | 5912 | 0.866 | 0.736 | 0.648 | 0.339 | 20.7 | 19.4 | 4.53 | 1.28 | 74% |
| M1 strips S=1 | 11823 | 0.866 | 0.737 | 0.650 | 0.346 | 20.7 | 19.4 | 4.59 | 1.33 | 75% |
| M2 Kalman fusion | 11823 | 0.893 | 0.742 | 0.543 | 0.315 | 17.3 | 19.0 | 3.20 | 1.21 | 73% |
| M3 Viterbi decode | 11823 | 0.902 | 0.748 | 0.546 | 0.315 | 16.5 | 18.9 | 2.87 | 1.20 | 72% |
| M4 particle filter @1.2kHz | 1182 | 0.901 | 0.750 | 0.550 | 0.328 | 16.5 | 18.8 | 2.05 | 0.95 | 70% |
| M4 particle filter @11.8kHz | 11823 | 0.906 | 0.752 | 0.546 | 0.340 | 16.0 | 18.7 | 1.76 | 1.06 | 70% |
| M4 PF learned likelihood | 1182 | 0.903 | 0.751 | 0.550 | 0.332 | 16.3 | 18.7 | 1.88 | 0.98 | 70% |
| M5 batch MAP smoother | 11823 | 0.901 | 0.747 | 0.546 | 0.315 | 16.5 | 18.9 | 2.88 | 1.20 | 72% |

## Rate-decimation curve (M3)

| output rate (Hz) | r dot x | prec x (') |
|---|---|---|
| 30 | 0.904 | nan |
| 100 | 0.901 | nan |
| 303 | 0.900 | 3.32 |
| 985 | 0.900 | 3.66 |
| 2956 | 0.899 | 3.79 |
| 11823 | 0.899 | 3.84 |

## Sub-frame validity (is the kHz content real?)

The 0.2 Hz dot cannot distinguish 15 Hz from 12 kHz tracking, so r-vs-dot alone does not prove kHz content. Two independent checks:

- **Independent-estimator band agreement (8-300 Hz, above the frame chain's 7.3 Hz Nyquist)**: M1 strips (joint 2D matchTemplate) vs M3 (per-line engine + Viterbi) agree at r = 0.574 (n = 90346), while each correlates with the interpolated 15 Hz chain at only r = 0.011 / 0.022. Agreement between independent measurement paths far above the chain baseline = genuine sub-frame signal.
- **Saccade physiology, m3_viterbi**: 1427 events (28.39/s), median amp 3.4', p90 14.2', main-sequence log-log slope 0.73 (corr 0.78).
- **Saccade physiology, m4_dpf_1182**: 1018 events (20.89/s), median amp 3.4', p90 8.7', main-sequence log-log slope 0.68 (corr 0.72).
- **Saccade physiology, m0_chain**: 0 events (0.00/s), median amp nan', p90 nan', main-sequence log-log slope nan (corr nan).

## Figures

- `khz2d_rate_accuracy.png` — accuracy / precision vs output rate (slow axis and 826 Hz alias gate marked).
- `khz2d_overlay.png` — calibrated trajectory overlay vs the dot.

## Decision log

- **Anchor chain**: full-frame phase correlation (`data.frame_truth`) mislocks on banding/low-overlap frames (chain-vs-dot |r| ~ 0.62, and only ~0.46 agreement with the strip-median chain). REPLACED at harness level by the median of per-strip 2D matchTemplate shifts (chain-vs-dot |r| ~ 0.90). Every method inherits this anchor.
- **Clock offset**: r against the 0.2 Hz dot loses ~0.1 per 100 ms of OFF error; OFF is therefore estimated once on a 25 ms grid from the raw per-line series and frozen for all methods (fair comparison).
- **M2 Kalman tuning**: quality-margin-scaled measurement noise (sigma/(q-q0)) and reseed-counts-as-update both REJECTED (each degraded r by 0.05-0.3 by trusting stale/garbage reacquisitions); final: sigma/q scaling, reacquire only from strong (q>0.45) recent measurements, reacquisitions don't count as valid updates.
- **M5 init**: initialised from the M3 Viterbi path; from-scratch initialisation converges to the same solution but slower. M5 ~= M3 on this data (the profile ridge is already globally consistent; the extra dynamics/anchor terms change little), so the cheap Viterbi is preferred operationally.
- **Vertical channel**: the per-line 1D NCC vertical residual is intrinsically noisier than horizontal strip matching (r_y raw ~0.38); fused (Kalman) vertical reaches the frame-chain ceiling ~0.75. Vertical is the weak axis of an x-scan system, as expected from first principles (a horizontal line constrains vertical only through appearance change).
