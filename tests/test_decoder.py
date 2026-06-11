"""G3 acceptance tests for the frozen differentiable decoder."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import decoder  # noqa: E402


def _atlas(seed=0, H=120, W=200):
    rng = np.random.default_rng(seed)
    # smooth-ish structured field so bilinear interpolation is meaningful
    base = rng.standard_normal((H, W))
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(base, 2.0).astype(np.float64)


def test_render_matches_numpy_bruteforce():
    A = _atlas()
    for perp, along, step in [(40.3, 12.7, 1.0), (60.0, 0.0, 1.5), (10.9, 50.2, 0.7)]:
        L = 80
        t = decoder.render(perp, along, A, L, col_step=step).detach().numpy()
        ref = decoder.render_numpy(perp, along, A, L, col_step=step)
        err = np.abs(t - ref).max()
        assert err < 1e-4, f"render vs numpy max err {err} at ({perp},{along},{step})"


def test_no_learned_parameters():
    # decoder module exposes only functions; render output is a pure function of inputs
    A = _atlas()
    l1 = decoder.render(30.0, 10.0, A, 50)
    l2 = decoder.render(30.0, 10.0, A, 50)
    assert torch.allclose(l1, l2)


def test_gradients_flow_and_match_finite_difference():
    A = torch.tensor(_atlas(seed=3), dtype=torch.float64)
    L = 64
    perp = torch.tensor(55.4, dtype=torch.float64, requires_grad=True)
    along = torch.tensor(33.1, dtype=torch.float64, requires_grad=True)
    line = decoder.render(perp, along, A, L)
    # scalar objective with structure so both grads are non-trivial
    w = torch.linspace(0.2, 1.3, L, dtype=torch.float64)
    obj = (line * w).sum()
    obj.backward()
    gp, ga = perp.grad.item(), along.grad.item()
    assert np.isfinite(gp) and np.isfinite(ga)
    assert abs(gp) > 0 and abs(ga) > 0

    def f(p, a):
        ln = decoder.render(torch.tensor(p, dtype=torch.float64),
                            torch.tensor(a, dtype=torch.float64), A, L)
        return float((ln * w).sum())

    eps = 1e-4
    fd_p = (f(55.4 + eps, 33.1) - f(55.4 - eps, 33.1)) / (2 * eps)
    fd_a = (f(55.4, 33.1 + eps) - f(55.4, 33.1 - eps)) / (2 * eps)
    assert abs(gp - fd_p) < 1e-3, f"perp grad {gp} vs fd {fd_p}"
    assert abs(ga - fd_a) < 1e-3, f"along grad {ga} vs fd {fd_a}"


def test_batched_render():
    A = _atlas()
    perp = torch.tensor([10.0, 20.5, 30.0])
    along = torch.tensor([5.0, 5.0, 5.0])
    lines = decoder.render(perp, along, A, 40)
    assert lines.shape == (3, 40)
    # each batch row matches the scalar render
    for i in range(3):
        single = decoder.render(perp[i].item(), along[i].item(), A, 40)
        assert torch.allclose(lines[i], single)


def test_render_on_real_atlas_row():
    """A line rendered at integer (perp,along)=(r,0) equals that atlas row slice."""
    A = _atlas(seed=7, H=80, W=150)
    r = 25
    L = 100
    line = decoder.render(float(r), 0.0, A, L).detach().numpy()
    assert np.abs(line - A[r, :L]).max() < 1e-9
