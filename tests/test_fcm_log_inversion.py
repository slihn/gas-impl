# Accuracy map for FracChiMean.pdf_by_log_inversion: the exact FCM pdf by Fourier inversion of the
# characteristic function of log V, in centred log space.
#
# Why this file matters: the library pdf() (frac_gamma) is only ~1e-5 relative, and fails outright
# from |k| ~ 50 (alpha 0.3) to ~300 (alpha 1.6). Every guard below came from a bug hit while building
# the inversion; each would silently return a plausible-looking but wrong density.

import numpy as np
import mpmath as mp
import pytest
from scipy.special import digamma

from .fcm_dist import FracChiMean, loggamma_diff


def _f_u_mpmath(alpha, k, u, theta=0.0, t_max_sd=300):
    """Density of U = log V - E[log V], fully independent of the double-precision code: mpmath
    log-gammas, digamma and quadrature, and its OWN cutoff (300 sd^-1, beyond the ~135 needed at
    alpha = 1.9, k = 2), so a bug in FracChiMean._t_max cannot also weaken the reference."""
    e = 1 / mp.mpf(alpha)
    g = (mp.mpf(alpha) - theta) / (2 * mp.mpf(alpha))
    sg = 1 if k > 0 else -1
    z = mp.mpf(k) - 1 if k > 0 else mp.mpf(-k)
    A = lambda x: mp.loggamma(e * x) - mp.loggamma(g * x)
    k1 = sg * (e * mp.digamma(z * e) - g * mp.digamma(g * z))
    sd = mp.sqrt(e**2 * mp.polygamma(1, z * e) - g**2 * mp.polygamma(1, g * z))
    phi = lambda t: mp.re(mp.exp(A(z + sg * 1j * t) - A(z) - 1j * t * k1 - 1j * t * u))
    return float(mp.quad(phi, mp.linspace(0, t_max_sd / sd, 61)) / mp.pi)


@pytest.mark.parametrize("alpha, k", [
    (0.3, 2.0), (0.3, 300.0), (0.3, -300.0),     # small alpha: library fails here at large |k|
    (0.5, 12.0),
    (1.2, 2.0), (1.2, -2.0),
    (1.9, 2.0), (1.9, -2.0), (1.9, 12.0),        # alpha -> 2, small |k|: phi decays slowly, log V one-sided
    (1.9, 300.0),
])
def test_log_inversion_matches_independent_mpmath(alpha, k):
    """Fails when the inversion's accuracy degrades: log-gamma cancellation at large |k|, a cutoff too
    short for the slow phi decay at alpha -> 2 (this gave 100% errors before the adaptive cutoff), or
    the branch sign. Error is scored against the peak, plus relative error where the density is
    resolvable in double precision (f > 1e-9 peak); deep one-sided tails (f ~ 1e-18 peak) are not."""
    mp.mp.dps = 30
    f = FracChiMean(alpha, k)
    sd = np.sqrt(f.log_var())
    u = np.array([-3.0, -1.5, 0.0, 1.5, 3.0]) * sd
    ours = f.log_v_pdf(u)
    ref = np.array([_f_u_mpmath(alpha, k, mp.mpf(ui)) for ui in u])
    peak = ref.max()
    assert np.max(np.abs(ours - ref)) / peak < 1e-11
    ok = ref > 1e-9 * peak
    assert np.max(np.abs(ours[ok] / ref[ok] - 1)) < 1e-8


@pytest.mark.parametrize("alpha, k", [(0.004, 1000.0), (0.004, -1000.0), (0.0004, 10000.0), (0.0004, -10000.0)])
def test_log_inversion_reproduces_exact_moments_at_large_k(alpha, k):
    """Fails when large-|k| precision is lost, where the library pdf cannot run at all: the
    lnGamma(x+h) - lnGamma(x) difference at x ~ 1e7 (cost 3e-8 before loggamma_diff), or numpy's
    naive complex log1p. Checks the density's mass and E[e^U], whose exact value is
    exp(K(1) - K'(0)) from the closed-form Mellin moment, and that the log_x entry point handles x
    beyond float range (E[log V] ~ 1380 at k = 1000)."""
    f = FracChiMean(alpha, k)
    sd = np.sqrt(f.log_var())
    u = np.linspace(-8 * sd, 8 * sd, 4001)
    fu = f.log_v_pdf(u)
    du = u[1] - u[0]
    z, sg = f._log_inversion_branch()
    k1 = sg * (f.eps * digamma(z * f.eps) - f.g * digamma(f.g * z))
    exact_E_eU = float(np.real(np.exp(f._log_cgf(1.0) - k1)))
    assert abs(np.sum(fu) * du - 1) < 1e-10
    assert abs(np.sum(np.exp(u) * fu) * du / exact_E_eU - 1) < 1e-9
    log_x = f.log_mean()                                       # |log_x| > 709 here: x is not a float
    assert np.isclose(f.log_pdf_by_log_inversion(log_x), np.log(fu[len(u) // 2]) - log_x, rtol=0, atol=1e-10)


@pytest.mark.parametrize("alpha, k, theta", [
    (0.6, 6.0, 0.0), (0.36364, 10.0, 0.0), (0.8, -5.0, 0.0), (0.33333, -10.0, 0.0),
    (0.9, 4.5, 0.3), (1.2, -4.0, -0.4), (1.5, 3.0, 0.2),
])
def test_log_inversion_agrees_with_library_pdf(alpha, k, theta):
    """Fails when the inversion computes a different FCM from the library's: a wrong g for theta != 0,
    a wrong branch sign (fcm_moment scales by sigma^n for k > 0 but sigma^-n for k < 0), or a wrong
    scale. The mpmath test shares the derivation of K, so only the library can catch a convention
    error. Tolerance is the library's own accuracy (~1e-5 relative), not the inversion's."""
    f = FracChiMean(alpha, k, theta)
    c, sd = f.log_mean(), np.sqrt(f.log_var())
    for z in (-2.0, -1.0, 0.0, 1.0, 2.0):
        x = np.exp(c + z * sd)
        assert np.isclose(f.pdf_by_log_inversion(x), float(f.pdf(x)), rtol=1e-4, atol=0)


@pytest.mark.parametrize("x", [20.0, 1e2, 1e4, 2.5e7, 1e9, 1e12])
def test_loggamma_diff_has_no_cancellation_at_large_x(x):
    """Fails when lnGamma(x + h) - lnGamma(x) is formed by subtracting two ~1e8-1e13 log-gammas, or
    when numpy's naive complex log1p is used for tiny h/x: both lose ~1e-8 to 1e-9 at large x."""
    mp.mp.dps = 50
    for h in (0.5j, 37j, 150j, -4.0, 1.0, 3 + 5j, 1e-6j):
        ref = complex(mp.loggamma(mp.mpf(x) + mp.mpc(h)) - mp.loggamma(mp.mpf(x)))
        assert abs(complex(loggamma_diff(x, h)) - ref) / max(1.0, abs(ref)) < 1e-14
