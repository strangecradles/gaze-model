# Ashton3 oculomotor video velocity check

Measured from the exact trace used by `people_data_fov_anim.py --person Ashton3` after the
2026-06-17 event-smoothing update (`VEL_SMOOTH_MS = 5 ms`).

## Verdict

The refreshed canonical video is physiologically plausible as a display trajectory:

- Final plotted line-rate max speed: **117.4 deg/s**
- Final plotted line-rate 99.9th percentile speed: **18.8 deg/s**
- Final plotted 30 fps max frame step: **0.36 deg**
- Final plotted 30 fps max speed: **10.85 deg/s**
- p1-p99 plotted range: **1.48 deg horizontal**, **1.67 deg vertical**

The apparent "few degree" motion is the pursuit trajectory scale, not a frame-to-frame jitter jump.
The largest marker jump in the rendered video is <0.4 deg. The previous 2 ms event-preserving
setting allowed one-line alias edges through as tiny saccade-like events; those reached ~386 deg/s
after display scaling. Increasing the event smoothing to 5 ms reduced that to ~117 deg/s while
keeping r-vs-dot unchanged (`x=0.94`, `y=0.95`).

## Files

- Canonical refreshed video: `people_data_results/gaze2d_Ashton3_m4_dpf.mp4`
- Raw diagnostic backup: `people_data_results/gaze2d_Ashton3_m4_dpf_raw.mp4`
- Machine-readable metrics: `results/ashton3_oculomotor_velocity.json`

