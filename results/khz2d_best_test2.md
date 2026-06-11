# kHz 2D Gaze on the TRUE x-Scan (test2) + 2D SLO — Deployment (Testbed B)

**Inputs**: test2 x-scan (11985 Hz line rate, processed in 12-sweep blocks -> 999 Hz effective) + the SAME-SESSION test1 mosaic as the 2D SLO. Absolute 2D anchors are computed at 20 Hz ONLY (the stated slow-axis budget): ~0.6 s coarse-band matches against the full mosaic.

**Channels**: horizontal gaze = complementary fusion (crossover 2 s) of the 20 Hz absolute anchors (low-pass: carries the pursuit band) and the trusted whole-line NCC along-shift cumsum (high-pass: the kHz sub-anchor content, drift dies in the high-pass); vertical gaze = per-block fine-band appearance match over mosaic rows (aliased/multimodal) resolved by each method between anchors.

**Caveat**: test2 has no co-registered raster, so unlike testbed A there is no frame-rate 2D registration truth; validation is against the pursuit dot (target, not the eye) and the ~32.5 Hz machine tracker. Self-consistency is necessary, not sufficient (see results/real_validation.md).

## Results

| method | rate (Hz) | r dot x | r dot y | r trk x | r trk y | RMS x (') | RMS y (') | valid |
|---|---|---|---|---|---|---|---|---|
| B0 anchors only | 20 | 0.233 | 0.049 | 0.066 | 0.031 | 46.0 | 28.6 | 100% |
| B-KF kalman | 999 | 0.454 | 0.231 | 0.272 | 0.259 | 42.5 | 27.6 | 97% |
| B-VIT viterbi | 999 | 0.445 | 0.211 | 0.263 | 0.237 | 42.3 | 28.0 | 100% |
| B-DPF particle filter | 999 | 0.462 | 0.320 | 0.288 | 0.178 | 42.5 | 27.1 | 90% |

## Per-stimulus-phase breakdown (B-KF, |r|)

Full-trace r under-states each axis (during H_sine the dot has no vertical motion, etc.). Per phase, against the driven axis:

| phase | r dot x | r dot y | r trk x | r trk y |
|---|---|---|---|---|
| H_sine | 0.84 | - | 0.60 | 0.26 |
| V_sine | - | 0.45 | 0.28 | 0.56 |
| circle | 0.36 | 0.20 | 0.42 | 0.11 |
| lissajous | 0.68 | 0.16 | 0.49 | 0.18 |

Note the along (x) channel is cross-coupled to vertical gaze: a vertical eye movement changes which mosaic row the line samples, and the locally-diagonal mosaic structure shifts the matched column with it (during V_sine the raw channel tracked dot_y at |r| ~ 0.8 with zero horizontal dot motion). A self-supervised leak fit (along vs anchor-row, no reference data) removes the common component and is included in the channel above; the residual sign inconsistency across phases (circle flips vs H_sine/lissajous) shows the leak is position-dependent — a joint 2D row+column readout per block is the next lever for testbed B horizontal.

- anchors: q med 0.70, good 1388/1391; per-block fine match q med 0.18 (vs ~0.2 cross-session in G15 — native scale pays).
- figure: `khz2d_test2.png`.

## Verdict — does the kHz channel add real 2D content?

On the honest joint clock offset (OFF = 2.05 s, see below), the vertical axis is the discriminator. The 20 Hz absolute-anchor baseline carries almost no vertical signal (B0: r dot y = 0.049, r trk y = 0.031); the per-block kHz appearance match resolved between anchors lifts it substantially (B-KF: r dot y = 0.231, r trk y = 0.259). Because the machine tracker is an INDEPENDENT measurement path (not the pursuit target), the rise in r trk y from the 20 Hz baseline to the kHz reconstruction is evidence of genuine sub-anchor vertical gaze content from the true x-scan + 2D SLO alone.

Horizontal (along): r dot x = 0.454 full-trace with the complementary-fusion channel (20 Hz anchor low-pass + trusted kHz NCC high-pass; was ~0.24 with the per-block rd_along KF — the per-block along match at ~3.8 px/deg is too low-SNR and was dragging the channel down). The pursuit band is carried by the smoothed absolute anchors; the kHz trusted channel adds the fast sub-anchor content (it measurably improves the independent-tracker corr and is neutral for the 0.2 Hz dot). Horizontal remains the weak axis of a horizontal-line scanner — along motion barely displaces the line.

## Decision log — clock offset (OFF)

- A single-axis along-vs-dot_x scan is **not** robust on this stimulus: the pursuit is a sequence of per-axis sinusoids (sync, H_sine, V_sine, circle, lissajous), so along-vs-dot_x has strong false maxima at H_sine-half-period aliases (we observed candidates near 11.3 / 17.2 / 19.4 s; an earlier pass mistakenly pinned OFF ~ 19.3 s off one such alias).

- `off_b` now maximises the **two-axis** score |r(along, dot_x)| + |r(anchor_row, dot_y)| over the full trace. A true offset must align BOTH axes with their own stimulus at once; aliases only line up one axis. The circle phase (32.5–48.5 s, both axes moving) is the tiebreaker: at OFF=2.05 the natural pairing holds (along·x=+0.31, row·y=+0.54, cross terms ~0) while the 17.2 s alias collapses there (along·x=-0.23, row·y=-0.17). The recovered OFF ~ 2 s is consistent with the mp4 creation_time lead (~2.2–2.6 s).

