# Tests for the Clark (1973) lognormal-normal distribution, lnn_dist.py.
#
# The references are independent of the double-precision code: mpmath at 50 digits over a wide,
# finely segmented window around the integrand's peak. (With coarse segments mpmath itself was off
# by 9e-6 at 20 sd and 1.5e-7 at 40 sd for s = 0.1, so the reference resolution matters.)

import numpy as np
import mpmath as mp
import pytest
from scipy import stats
from scipy.integrate import quad

from .lnn_dist import (lnn, lnn_pdf_by_gauss_hermite, lnn_cdf_by_gauss_hermite, lnn_log_pdf_by_gauss_hermite,
                       lnn_var, lnn_ex_kurt, lnn_spd, lnn_s_from_ex_kurt, _lnn_pdf_peak,
                       lnn_from_spd_to_ex_kurt, lnn_from_ex_kurt_to_spd)

S_GRID = [0.1, 0.5, 1.0, 1.5]
Z_SD = np.array([0.0, 1.0, 8.0, 40.0])          # in units of sd = exp(s^2)


def _pdf_mp(z, s):
    c, w = (float(v) for v in _lnn_pdf_peak(z, s))
    zz, ss = mp.mpf(z), mp.mpf(s)
    g = lambda n: mp.exp(-n**2 / 2 - ss * n - zz**2 * mp.exp(-2 * ss * n) / 2) / (2 * mp.pi)
    return mp.quad(g, mp.linspace(c - 60 * w, c + 60 * w, 241))


def _cdf_mp(z, s):                               # z <= 0
    c, w = (float(v) for v in _lnn_pdf_peak(z, s))
    zz, ss = mp.mpf(z), mp.mpf(s)
    g = lambda n: mp.npdf(n) * mp.ncdf(zz * mp.exp(-ss * n))
    return mp.quad(g, mp.linspace(c - 60 * w, max(c + 60 * w, 40.0), 241))


@pytest.mark.parametrize("s", S_GRID)
def test_pdf_matches_mpmath_into_the_tails(s):
    """Fails when the pdf loses the mass in the tails (a Gauss-Hermite rule not centred at the
    integrand's peak, or quad not split there) or when 64 nodes return at large s (1e-10 at s = 1.5)."""
    mp.mp.dps = 50
    z = Z_SD * np.exp(s * s)
    ref = np.array([float(_pdf_mp(zi, s)) for zi in z])
    assert np.max(np.abs(lnn.pdf(z, s) / ref - 1)) < 5e-12                       # class: quad over lognorm
    assert np.max(np.abs(lnn_pdf_by_gauss_hermite(z, s) / ref - 1)) < 5e-12      # standalone


@pytest.mark.parametrize("s", S_GRID)
def test_cdf_matches_mpmath_in_the_left_tail_and_is_symmetric(s):
    """Fails when the left tail is formed as 1 - (something near 1), losing all relative accuracy,
    when the Newton centring of the cdf integrand breaks, or when the symmetry F(z) = 1 - F(-z) does."""
    mp.mp.dps = 50
    z = -Z_SD * np.exp(s * s)
    ref = np.array([float(_cdf_mp(zi, s)) for zi in z])
    assert np.max(np.abs(lnn.cdf(z, s) / ref - 1)) < 5e-12
    assert np.max(np.abs(lnn_cdf_by_gauss_hermite(z, s) / ref - 1)) < 5e-12
    zz = np.array([0.3, 1.7, 4.0])
    assert np.max(np.abs(lnn.cdf(zz, s) + lnn.cdf(-zz, s) - 1)) < 1e-14


@pytest.mark.parametrize("s", [0.25, 0.5, 1.0])
def test_pdf_reproduces_the_closed_form_moments_and_spd(s):
    """Fails when the pdf and the closed forms disagree on the parameterization (s versus Clark's
    sigma_1 = 2s, or a missing e^{-s n} Jacobian): mass, variance, 4th moment, SPD and the exact
    kurtosis-SPD relation 1 + gamma2/3 = (sqrt(2 pi) SPD)^(8/3) must all hold together."""
    f = lambda z: lnn_pdf_by_gauss_hermite(z, s)
    m0 = quad(f, -np.inf, np.inf, epsabs=1e-14, epsrel=1e-13, limit=500)[0]
    m2 = quad(lambda z: z**2 * f(z), -np.inf, np.inf, epsabs=1e-14, epsrel=1e-13, limit=500)[0]
    m4 = quad(lambda z: z**4 * f(z), -np.inf, np.inf, epsabs=1e-13, epsrel=1e-12, limit=500)[0]
    assert abs(m0 - 1) < 5e-12                                   # the outer quad's own error ~1e-12
    assert abs(m2 / lnn_var(s) - 1) < 1e-11
    assert abs((m4 / m2**2 - 3) / lnn_ex_kurt(s) - 1) < 1e-9
    spd = float(lnn.std(s) * lnn.pdf(0.0, s))
    assert abs(spd / lnn_spd(s) - 1) < 1e-13
    assert abs((np.sqrt(2 * np.pi) * spd)**(8 / 3) - (1 + lnn_ex_kurt(s) / 3)) < 1e-12
    assert abs(lnn_s_from_ex_kurt(lnn_ex_kurt(s)) - s) < 1e-14


def test_rvs_follows_the_cdf():
    """Fails when the sampler uses a different s than the density (e.g. Clark's sigma_1 = 2s):
    200k draws are tested against the analytic cdf."""
    s = 0.5
    x = lnn.rvs(s, size=200_000, random_state=np.random.default_rng(3))
    ks = stats.kstest(x, lambda z: lnn_cdf_by_gauss_hermite(z, s))
    assert ks.statistic < 0.004 and ks.pvalue > 1e-3


def test_log_pdf_at_extreme_z_uses_the_overflow_safe_peak():
    """Fails when z^2 is formed anywhere (it overflows past z ~ 1e154): in the Lambert-W argument
    2 s^2 z^2 e^{2 s^2} or in the integrand's z^2 e^{-2 s n}. This returned nan before the fix. The log
    pdf (the pdf itself underflows) must still match mpmath."""
    mp.mp.dps = 50
    s, z = 0.5, 1e200
    c, w = (float(v) for v in _lnn_pdf_peak(z, s))
    zz, ss = mp.mpf(z), mp.mpf(s)
    g = lambda n: mp.exp(-n**2 / 2 - ss * n - zz**2 * mp.exp(-2 * ss * n) / 2) / (2 * mp.pi)
    ref = float(mp.log(mp.quad(g, mp.linspace(c - 60 * w, c + 60 * w, 241))))
    assert abs(float(lnn_log_pdf_by_gauss_hermite(z, s)) - ref) < 1e-10 * abs(ref)


@pytest.mark.parametrize("s", [1e-4, 0.1, 0.5, 1.0, 1.5])
def test_spd_kurtosis_relation_matches_the_lnn_shape(s):
    """Fails when the kurtosis-SPD conversions drift from the LNN they describe (a wrong exponent, 8/3
    versus 3/8, or a dropped sqrt(2 pi)). SPD -> kurtosis is ill-conditioned near the Gaussian point:
    a relative error eps in the SPD becomes an ABSOLUTE error ~8 eps in ex_kurt, while ex_kurt ~ 12 s^2
    is tiny there (relative error 2e-9 at s = 1e-4), so that direction is checked in absolute terms."""
    assert abs(lnn_from_ex_kurt_to_spd(lnn_ex_kurt(s)) / lnn_spd(s) - 1) < 1e-14
    assert abs(lnn_from_spd_to_ex_kurt(lnn_spd(s)) - lnn_ex_kurt(s)) < 1e-12 * (1 + lnn_ex_kurt(s))
