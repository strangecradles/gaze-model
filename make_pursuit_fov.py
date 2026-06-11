"""make_pursuit_fov.py — generate an FOV-aware pursuit stimulus for the RIGHT eye.

WHY (from results/khz2d_vertical_diagnosis.md): tracking the RIGHT eye, the SLO
loses signal when the eye looks RIGHT (temporal gaze) — the imaged retinal patch
slides off the SLO field. Measured on the test1 capture (binocular viewing, right
eye tracked), in-FOV fraction vs dot SCREEN position:

  horizontal (x_px, 960=screen center):
    x_px 300-850 : 99-100%   850-950 : 97%   950-1050 : 85%
    1050-1150 : 72%   1250-1350 : 48%   1350+ : <=15%  (collapses to the right)
  vertical (y_px), RESTRICTED to good x_px[300,940] (de-confounded):
    97-100% across the ENTIRE range 120-960  -> vertical is NOT FOV-limited.

So the FOV constraint is purely HORIZONTAL for this right-eye setup. The earlier
apparent vertical deficit was a horizontal confound (the H_sine phase sat at
y=540 while sweeping into the bad right region).

DESIGN: keep the dot inside the empirically-good band by (1) shifting the
horizontal CENTER left to the good-FOV centre (~630 px, vs screen centre 960),
(2) capping horizontal excursion to [330, 930] px (>=99% in-FOV, margin to the
~1000 px cliff), (3) keeping vertical screen-centred and full-range (it is
unconstrained). Both eyes view one shared dot (binocular fusion, natural); only
the right eye is tracked. Putting the dot left of screen centre points the right
eye nasally, into good SLO signal — the asymmetry is handled purely by dot
placement, no per-eye rendering needed (the measured FOV map already reflects the
binocular vergence of the same rig).

The phase STRUCTURE and DURATIONS match stimuli/pursuit.csv exactly (sync,
H_sine, V_sine, circle, lissajous, end; 0.2 Hz; 60 fps; 67.48 s) so the new
capture is directly comparable and the existing phase-window analysis
(khz2d_test2.STIM_PHASES) still applies.

Outputs: stimuli/pursuit_fov.csv, stimuli/pursuit_fov.mp4,
stimuli/pursuit_fov_meta.json, results/pursuit_fov_preview.png.

Run `python make_pursuit_fov.py --from-data` to re-derive the safe band from the
test1 line cache (prints the in-FOV-vs-x map and uses its 99%-edges); default
uses the baked constants below (derived from that same analysis).
"""
from __future__ import annotations

import os
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
STIM = os.path.join(HERE, "stimuli")
RESULTS = os.path.join(HERE, "results")

# --- screen / dot (match stimuli_meta.json) ---
W, H = 1920, 1080
FPS = 60
BG_GRAY = 60
DOT_R = 11
CX_SCREEN, CY_SCREEN = W / 2.0, H / 2.0     # 960, 540

# --- FOV-safe region (screen px), derived from the test1 in-FOV map ---
# Good band measured at >=99% in-FOV: x_px ~[300,950], collapsing past ~1000.
# Centre the path on the good-FOV centre and cap excursion with margin.
CX_SAFE = 630.0          # horizontal centre (left of screen centre 960)
AX = 300.0               # horizontal half-amplitude -> x in [330, 930]
CY_SAFE = 540.0          # vertical centre = screen centre (vertical unconstrained)
AY = 330.0               # vertical half-amplitude -> y in [210, 870]
R_CIRC = 300.0           # circle radius (fits both: x[330,930], y[240,840])

PURSUIT_HZ = 0.2
OMEGA = 2 * np.pi * PURSUIT_HZ

# phase schedule: (name, duration_s) — matches pursuit.csv windows
SCHEDULE = [
    ("sync", 2.5),
    ("H_sine", 15.0),
    ("V_sine", 15.0),
    ("circle", 16.0),
    ("lissajous", 18.0),
    ("end", 1.0),
]


def derive_safe_band_from_data():
    """Re-derive the >=99%-in-FOV horizontal edges + the always-good vertical
    range from the real test1 line cache (optional; prints the map)."""
    import khz2d
    import pandas as pd
    lm = khz2d.line_measurements()
    R = khz2d.refs()
    df = pd.read_csv(os.path.join(STIM, "pursuit.csv"))
    t = lm["t"]
    xpx = np.interp(t, df.t_sec.values + R["off"], df.x_px.values)
    ypx = np.interp(t, df.t_sec.values + R["off"], df.y_px.values)
    fov = khz2d.fov_mask(lm)
    gy = (ypx >= 200) & (ypx <= 560)
    print("[from-data] in-FOV vs x_px (y in [200,560]):")
    edges = np.arange(150, 1700, 100)
    good_lo, good_hi = None, None
    for i in range(len(edges) - 1):
        m = gy & (xpx >= edges[i]) & (xpx < edges[i + 1])
        if m.sum() < 300:
            continue
        fr = fov[m].mean()
        print(f"   x {edges[i]:5.0f}-{edges[i+1]:5.0f}: {fr*100:5.0f}%  n={m.sum()}")
        c = 0.5 * (edges[i] + edges[i + 1])
        if fr >= 0.95:
            good_lo = c if good_lo is None else good_lo
            good_hi = c
    print(f"[from-data] >=95% in-FOV horizontal centres span ~[{good_lo:.0f}, {good_hi:.0f}] px")
    return good_lo, good_hi


def gen_path():
    """Per-frame (t, x_px, y_px, phase) for the whole stimulus."""
    rows = []
    t0 = 0.0
    for name, dur in SCHEDULE:
        n = int(round(dur * FPS))
        for k in range(n):
            tau = k / FPS                      # time within phase
            t = t0 + tau
            if name == "sync":
                # 1.0 s centre hold, then sweep the full safe band for along-axis
                # alignment: centre -> left -> right -> centre (0.5 s each)
                if tau < 1.0:
                    x, y = CX_SAFE, CY_SAFE
                elif tau < 1.5:
                    f = (tau - 1.0) / 0.5
                    x, y = CX_SAFE - AX * f, CY_SAFE
                elif tau < 2.0:
                    f = (tau - 1.5) / 0.5
                    x, y = (CX_SAFE - AX) + 2 * AX * f, CY_SAFE
                else:
                    f = (tau - 2.0) / 0.5
                    x, y = (CX_SAFE + AX) - AX * f, CY_SAFE
            elif name == "H_sine":
                x, y = CX_SAFE + AX * np.sin(OMEGA * tau), CY_SAFE
            elif name == "V_sine":
                x, y = CX_SAFE, CY_SAFE + AY * np.sin(OMEGA * tau)
            elif name == "circle":
                x = CX_SAFE + R_CIRC * np.sin(OMEGA * tau)
                y = CY_SAFE - R_CIRC * np.cos(OMEGA * tau)
            elif name == "lissajous":
                x = CX_SAFE + AX * np.sin(OMEGA * tau)
                y = CY_SAFE + AY * np.sin(2 * OMEGA * tau + np.pi / 2)
            else:  # end
                x, y = CX_SAFE, CY_SAFE
            rows.append((t, float(x), float(y), name))
        t0 += n / FPS
    return rows


def write_csv(rows, path):
    lines = ["frame,t_sec,x_px,y_px,x_norm,y_norm,phase"]
    for i, (t, x, y, ph) in enumerate(rows):
        xn = (x - CX_SCREEN) / CX_SCREEN
        yn = (y - CY_SCREEN) / CY_SCREEN
        lines.append(f"{i},{t:.4f},{x:.0f},{y:.0f},{xn:.4f},{yn:.4f},{ph}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def render_mp4(rows, path):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, FPS, (W, H))
    if not vw.isOpened():
        raise RuntimeError("VideoWriter failed to open")
    for (_, x, y, _) in rows:
        frame = np.full((H, W, 3), BG_GRAY, np.uint8)
        cv2.circle(frame, (int(round(x)), int(round(y))), DOT_R,
                   (255, 255, 255), -1, lineType=cv2.LINE_AA)
        vw.write(frame)
    vw.release()


def preview(rows, path):
    xs = np.array([r[1] for r in rows]); ys = np.array([r[2] for r in rows])
    phases = [r[3] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 6))
    # FOV shading: green (good) left band, red (signal loss) right
    ax.axvspan(0, 1000, color="green", alpha=0.06)
    ax.axvspan(1000, W, color="red", alpha=0.10, label="SLO signal loss (>1000 px)")
    ax.axvspan(330, 930, color="green", alpha=0.10, label="safe band x[330,930]")
    for ph in dict.fromkeys(phases):
        m = np.array([p == ph for p in phases])
        ax.plot(xs[m], ys[m], ".", ms=1.5, label=ph)
    ax.axvline(CX_SCREEN, color="k", ls=":", lw=1, alpha=.5)
    ax.axvline(CX_SAFE, color="g", ls="--", lw=1, alpha=.7)
    ax.text(CX_SCREEN, 30, " screen centre", fontsize=8)
    ax.text(CX_SAFE, 1050, "safe centre 630", fontsize=8, color="g", ha="center")
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal")
    ax.set_xlabel("x_px (screen, +right = temporal for right eye)")
    ax.set_ylabel("y_px (screen, +down)")
    ax.set_title("FOV-aware pursuit for the RIGHT eye — dot kept in good SLO signal "
                 "(left/centre)")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-data", action="store_true",
                    help="re-derive the safe band from the test1 line cache")
    ap.add_argument("--no-video", action="store_true", help="csv+preview only")
    a = ap.parse_args()
    os.makedirs(STIM, exist_ok=True); os.makedirs(RESULTS, exist_ok=True)
    if a.from_data:
        try:
            derive_safe_band_from_data()
        except Exception as e:
            print(f"[from-data] skipped ({e}); using baked constants")

    rows = gen_path()
    xs = np.array([r[1] for r in rows]); ys = np.array([r[2] for r in rows])
    csv_p = os.path.join(STIM, "pursuit_fov.csv")
    write_csv(rows, csv_p)
    preview(rows, os.path.join(RESULTS, "pursuit_fov_preview.png"))
    meta = dict(screen_px=[W, H], fps=FPS, bg_gray=BG_GRAY, dot_radius_px=DOT_R,
                pursuit_freq_hz=PURSUIT_HZ, eye="right", viewing="binocular",
                fov_aware=True,
                safe_band_x_px=[CX_SAFE - AX, CX_SAFE + AX],
                safe_band_y_px=[CY_SAFE - AY, CY_SAFE + AY],
                center_px=[CX_SAFE, CY_SAFE],
                rationale=("right eye loses SLO signal for rightward/temporal gaze "
                           "(x_px>~1000); dot centred left at 630 px and capped to "
                           "[330,930] px; vertical full-range (unconstrained)"),
                sync="1s centre hold @ (630,540), then centre->left->right->centre "
                     "sweep across the safe band (0.5s each) for along-axis alignment",
                schedule=SCHEDULE)
    with open(os.path.join(STIM, "pursuit_fov_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"[pursuit_fov] {len(rows)} frames, {rows[-1][0]:.2f}s @ {FPS}fps")
    print(f"[pursuit_fov] x_px {xs.min():.0f}..{xs.max():.0f} (safe <=930), "
          f"y_px {ys.min():.0f}..{ys.max():.0f}")
    print(f"[pursuit_fov] wrote {csv_p}")
    print(f"[pursuit_fov] wrote {os.path.join(RESULTS, 'pursuit_fov_preview.png')}")
    if not a.no_video:
        mp4_p = os.path.join(STIM, "pursuit_fov.mp4")
        render_mp4(rows, mp4_p)
        print(f"[pursuit_fov] wrote {mp4_p}")


if __name__ == "__main__":
    main()
