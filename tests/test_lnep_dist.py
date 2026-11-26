# Tests for LNEP (lnep_dist.py): the exponential power kernel with lognormal volatility.
#
# The references are independent of the double-precision code: mpmath at 50 digits over a finely segmented window
# around the integrand's peak, Clark's LNN from lnn_dist at alpha = 2, scipy's gennorm at s = 0, and parameter
# recovery from simulated draws.

import numpy as np
import mpmath as mp
import pytest
from scipy import stats
from scipy.integrate import quad

from .lnep_dist import (lnep, lnep_pdf_by_trapezoid, lnep_cdf_by_trapezoid, lnep_var, lnep_ex_kurt, lnep_spd,
                        ep_ws, _ep_log_f0, _lnep_pdf_peak, LNEP_Fit)
from .lnn_dist import lnn

GRID = [(1.0, 0.5), (3.0, 0.1), (3.0, 1.0), (6.0, 0.5), (12.0, 1.0)]
Z_SD = np.array([0.0, 0.5, 1.0, 8.0, 40.0])     # in units of sd = exp(s^2)


def _ref(z, alpha, s, kind):
    c, w = (float(v) for v in _lnep_pdf_peak(z, alpha, s))
    zz, ss, aa = mp.mpf(z), mp.mpf(s), mp.mpf(alpha)
    ws, f0 = mp.mpf(float(ep_ws(alpha))), mp.e**mp.mpf(float(_ep_log_f0(alpha)))
    if kind == 'pdf':
        g = lambda n: mp.npdf(n) * mp.e**(-ss*n) * f0 * mp.e**(-(abs(zz*mp.e**(-ss*n))/ws)**aa)
        return mp.quad(g, mp.linspace(c - 60*w, c + 60*w, 241))
    F = lambda t: mp.gammainc(1/aa, (abs(t)/ws)**aa, mp.inf, regularized=True)/2      # EP cdf at t <= 0
    g = lambda n: mp.npdf(n) * F(zz*mp.e**(-ss*n))
    return mp.quad(g, mp.linspace(c - 60*w, max(c + 60*w, 40.0), 241))


@pytest.mark.parametrize("alpha, s", GRID)
def test_pdf_and_left_cdf_match_mpmath_into_the_tails(alpha, s):
    """Fails when a rule loses the mass in the tails (not centred at the integrand's peak, or quad not split there),
    or cannot resolve the kernel's wall exp(-c e^{-alpha s n}) at large alpha s (Gauss-Hermite gave 1e-4 at
    alpha = 12), or when the left tail is formed as 1 - (something near 1)."""
    mp.mp.dps = 50
    z = Z_SD*np.exp(s*s)
    rp = np.array([float(_ref(v, alpha, s, 'pdf')) for v in z])
    rc = np.array([float(_ref(-v, alpha, s, 'cdf')) for v in z])
    assert np.max(np.abs(lnep.pdf(z, alpha, s)/rp - 1)) < 1e-11                 # class: quad over lognorm
    assert np.max(np.abs(lnep_pdf_by_trapezoid(z, alpha, s)/rp - 1)) < 1e-11    # standalone
    assert np.max(np.abs(lnep.cdf(-z, alpha, s)/rc - 1)) < 1e-11
    assert np.max(np.abs(lnep_cdf_by_trapezoid(-z, alpha, s)/rc - 1)) < 1e-11
    zz = np.array([0.3, 1.7, 4.0])
    assert np.max(np.abs(lnep.cdf(zz, alpha, s) + lnep.cdf(-zz, alpha, s) - 1)) < 1e-14


def test_alpha_two_is_lnn_and_s_zero_is_the_exponential_power():
    z = np.array([0.0, 0.7, 3.0, 12.0])
    np.testing.assert_allclose(lnep.pdf(z, 2.0, 0.4), lnn.pdf(z, 0.4), rtol=1e-12)
    np.testing.assert_allclose(lnep_pdf_by_trapezoid(z, 2.0, 0.4), lnn.pdf(z, 0.4), rtol=1e-12)
    np.testing.assert_allclose(lnep_cdf_by_trapezoid(-z, 2.0, 0.4), lnn.cdf(-z, 0.4), rtol=1e-12)
    ws = float(ep_ws(3.0))
    np.testing.assert_allclose(lnep.pdf(z, 3.0, 0.0), stats.gennorm.pdf(z, 3.0, scale=ws), rtol=1e-12)
    np.testing.assert_allclose(lnep_cdf_by_trapezoid(-z, 3.0, 0.0), stats.gennorm.cdf(-z, 3.0, scale=ws), rtol=1e-12)


@pytest.mark.parametrize("alpha, s", [(4.0, 0.33), (3.0, 0.6), (1.5, 0.1)])
def test_closed_forms_match_the_density(alpha, s):
    """Variance, excess kurtosis (via _munp and _stats) and SPD against moments of the density itself."""
    f = lambda x: float(lnep_pdf_by_trapezoid(x, alpha, s))
    # symmetric: twice the half-line integral, which reaches the lognormal tail (a cut at 80 lost 3e-7 of m2 at s = 0.6)
    m = [2*(quad(lambda x: x**k*f(x), 0, 10, limit=500)[0] + quad(lambda x: x**k*f(x), 10, np.inf, limit=500)[0])
         for k in (0, 2, 4)]
    assert m[0] == pytest.approx(1.0, abs=1e-8)
    assert m[1] == pytest.approx(lnep_var(alpha, s), rel=1e-7)
    assert m[2] == pytest.approx(lnep.moment(4, alpha, s), rel=1e-6)
    _, var, skew, kurt = lnep.stats(alpha, s, moments='mvsk')
    assert (var, skew) == (pytest.approx(m[1], rel=1e-7), 0.0)
    assert kurt == pytest.approx(float(lnep_ex_kurt(alpha, s)), rel=1e-12)
    assert kurt == pytest.approx(m[2]/m[1]**2 - 3, rel=1e-6)
    assert f(0.0)*np.sqrt(m[1]) == pytest.approx(float(lnep_spd(alpha, s)), rel=1e-7)


def test_rvs_follow_the_cdf():
    x = lnep.rvs(3.0, 0.4, size=20000, random_state=np.random.default_rng(11))
    assert stats.kstest(x, lambda v: lnep_cdf_by_trapezoid(v, 3.0, 0.4)).pvalue > 0.01


def test_moment_inversion_round_trip_and_region():
    for alpha, s in [(3.0, 0.2), (4.0, 0.33), (1.5, 0.3), (8.0, 0.05)]:
        a, s_, status = LNEP_Fit.invert(*LNEP_Fit.v12(alpha, s))
        assert status == 'ok'
        assert (a, s_) == pytest.approx((alpha, s), rel=1e-8)
    # a gap v1 - v2 beyond the uniform-kernel limit, and a point left of the EP curve, are outside
    assert LNEP_Fit.invert(0.008, -0.1665)[2] == 'outside'
    assert LNEP_Fit.invert(-0.12, -0.15)[2] == 'outside'


def test_moments_fit_matches_the_sample_shape():
    x = LNEP_Fit.rvs_of(5000, 3.0, 0.25, 0.0, 0.01, rng=5)
    f = LNEP_Fit(x, method='moments')
    assert f.status == 'ok' and f.method == 'moments'
    assert LNEP_Fit.v12(f.alpha, f.s) == pytest.approx((f.shape['v1'], f.shape['v2']), abs=1e-10)
    assert (f.loc, f.sd) == (pytest.approx(x.mean()), pytest.approx(x.std()))


def test_mle_recovers_simulated_parameters_and_tops_the_profile():
    x = LNEP_Fit.rvs_of(20000, 3.0, 0.2, 0.0005, 0.011, rng=7)
    f = LNEP_Fit(x)
    assert f.status == 'ok'
    assert f.alpha == pytest.approx(3.0, abs=0.6)      # alpha and s trade off along a ridge, so alpha is the looser one
    assert f.s == pytest.approx(0.2, abs=0.05)
    assert f.sd == pytest.approx(0.011, rel=0.02)
    prof = f.profile_alpha([f.alpha, 2.0])
    assert prof.loglik.iloc[0] == pytest.approx(f.loglik(), abs=1e-4)
    assert prof.loglik.iloc[1] < f.loglik()
    assert LNEP_Fit.pdf_of(x[:5], 2.0, *prof.loc[2.0, ['s', 'loc', 'sd']]) == pytest.approx(
        np.exp(LNEP_Fit.logpdf_of(x[:5], 2.0, *prof.loc[2.0, ['s', 'loc', 'sd']])))
