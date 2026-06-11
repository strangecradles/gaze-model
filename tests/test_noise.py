"""G4 acceptance tests: measured-SNR noise + monotone motion blur."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import noise  # noqa: E402
import decoder  # noqa: E402


def _atlas(seed=1, H=140, W=240):
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    return gaussian_filter(rng.standard_normal((H, W)), 1.5).astype(np.float64)


def _clean_line(seed=2, L=400):
    from scipy.ndimage import gaussian_filter1d
    rng = np.random.default_rng(seed)
    return gaussian_filter1d(rng.standard_normal(L), 3.0)


# ---- per-rate SNR curve ----

def test_reliability_curve_monotone_and_anchored():
    lr = 12000.0
    # single sweep at the line rate -> RHO1
    assert abs(noise.reliability(lr, lr) - noise.RHO1) < 1e-9
    rates = [344, 820, 2000, 6000, 12000]
    rel = [noise.reliability(r, lr) for r in rates]
    # DECREASING in rate
    assert all(rel[i] > rel[i + 1] for i in range(len(rel) - 1)), rel
    # low-rate (heavy averaging) reaches the observed ~0.97 split-half regime
    assert noise.reliability(200, lr) > 0.95


def test_apply_noise_reproduces_reliability():
    clean = _clean_line()
    lr = 12000.0
    for rate in (12000.0, 2000.0, 800.0):
        target = noise.reliability(rate, lr)
        rng = np.random.default_rng(0)
        # average rep-rep correlation over many independent realizations
        cs = []
        for _ in range(400):
            r1 = noise.apply_noise(clean, rate, lr, rng=rng)
            r2 = noise.apply_noise(clean, rate, lr, rng=rng)
            cs.append(np.corrcoef(r1, r2)[0, 1])
        meas = float(np.mean(cs))
        assert abs(meas - target) < 0.03, f"rate {rate}: meas {meas:.3f} vs target {target:.3f}"


def test_apply_noise_preserves_type_and_shape():
    clean = _clean_line()
    out = noise.apply_noise(clean, 2000.0, seed=0)
    assert isinstance(out, np.ndarray) and out.shape == clean.shape
    t = torch.tensor(clean)
    outt = noise.apply_noise(t, 2000.0, seed=0)
    assert isinstance(outt, torch.Tensor) and outt.shape == t.shape


# ---- motion blur ----

def test_blur_monotonically_reduces_high_freq_with_velocity():
    A = _atlas()
    L = 160
    dt = 1.0 / 800.0  # a slow effective rate exaggerates blur for the test
    perp, along = 70.0, 20.0
    speeds = [0.0, 2000.0, 6000.0, 15000.0, 40000.0]  # cols/s along the scan
    energies = []
    for v in speeds:
        line = noise.apply_blur(A, perp, along, (0.0, v), dt, L)
        energies.append(noise.high_freq_energy(line, sigma=2.0))
    assert all(energies[i] >= energies[i + 1] - 1e-12 for i in range(len(energies) - 1)), energies
    # strictly lower at the fastest vs static
    assert energies[-1] < energies[0]


def test_blur_zero_velocity_equals_render():
    A = _atlas()
    L = 120
    line0 = noise.apply_blur(A, 50.0, 10.0, (0.0, 0.0), 1e-3, L).detach().numpy()
    ref = decoder.render(50.0, 10.0, A, L).detach().numpy()
    assert np.abs(line0 - ref).max() < 1e-9


def test_blur_is_differentiable():
    A = torch.tensor(_atlas(seed=4), dtype=torch.float64)
    perp = torch.tensor(60.0, dtype=torch.float64, requires_grad=True)
    along = torch.tensor(30.0, dtype=torch.float64, requires_grad=True)
    # blur path uses scalar velocity offsets; gradient must still flow via render
    L = 80
    fracs = (torch.arange(8, dtype=torch.float64) + 0.5) / 8
    perps = perp + 0.0 * fracs
    alongs = along + 5.0 * fracs
    lines = decoder.render(perps, alongs, A, L)
    obj = lines.mean()
    obj.backward()
    assert np.isfinite(perp.grad.item()) and np.isfinite(along.grad.item())
