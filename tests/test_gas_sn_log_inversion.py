# Tests for GAS_SN pdf_by_log_inversion, the pdf variant for low alpha and high k.
#
# Two independent references: the library _pdf where it is still accurate, and, where it is not, the
# exact moments (fcm_moment, gamma-function ratios -- no FCM pdf involved), integrated against the new pdf.

import numpy as np
import pytest

from .gas_sn_dist import GAS_SN
from .fcm_dist import FracChiMean


@pytest.mark.parametrize("alpha, k, beta", [(0.9, 3.13, -0.2), (0.7, 4.5, 0.5), (0.5, 6.0, -0.2), (1.2, -3.0, -0.4)])
def test_matches_library_pdf_where_the_library_is_accurate(alpha, k, beta):
    """Fails when the variant's units or skew differ from `_pdf`: v not scaled by e^{E log V}, beta's
    sign flipped, or a missing factor of v. Tolerance is the library's own accuracy here (~1e-5)."""
    g = GAS_SN(alpha=alpha, k=k, beta=beta, scale=1.7, loc=0.3)
    sd = np.sqrt(g.var())
    x = 0.3 + sd * np.array([-4.0, -1.0, 0.0, 0.5, 2.0])
    lib = np.array([g.pdf(float(v)) for v in x])
    assert np.max(np.abs(g.pdf_by_log_inversion(x) / lib - 1)) < 1e-4


@pytest.mark.parametrize("alpha, k", [(0.1, 26.0), (0.2, 12.9)])
def test_reproduces_exact_moments_where_the_library_pdf_fails(alpha, k):
    """Fails when the variant goes wrong at low alpha / high k, where `_pdf` is 100% off or zero: the
    mass, mean and second moment of the new pdf must equal the exact values."""
    beta = -0.2
    g = GAS_SN(alpha=alpha, k=k, beta=beta)
    e = np.exp(FracChiMean(alpha, k).log_mean())          # X is O(1/e) in library units; y = X e
    # composite 20-point Gauss-Legendre on |y| <= 1000 (sd ~ 1.5): +-200 still leaves 2e-11 of mass
    # out at alpha = 0.2. The pdf is called in chunks, each is an (n_x, 3001) array.
    gx, gw = np.polynomial.legendre.leggauss(20)
    edges = np.linspace(-1000.0, 1000.0, 2001)
    half = np.diff(edges)[:, None] / 2
    y = ((edges[:-1, None] + edges[1:, None]) / 2 + half * gx).ravel()
    w = (half * gw).ravel()
    fy = np.concatenate([g.pdf_by_log_inversion(c / e) / e for c in np.array_split(y, 20)])
    mass = np.sum(w * fy)
    mean = np.sum(w * y * fy) / e
    second = np.sum(w * y * y * fy) / e**2
    assert abs(mass - 1) < 1e-12
    assert abs(mean / g.mean() - 1) < 1e-10
    assert abs(second / (g.var() + g.mean()**2) - 1) < 1e-8


@pytest.mark.parametrize("alpha, k, beta", [(0.9, 3.13, -0.2), (0.3, 10.1, -0.12)])
def test_cdf_and_squared_quantiles_match_the_library_cdf(alpha, k, beta):
    """Fails when the cdf variant or the X^2 quantiles drift from the library cdf where that cdf is
    accurate. The X^2 check is the one FracF_PPF.ppf fails at alpha = 0.3, k = 10.1 (its median of X^2
    has a library probability of 0.90), so it must be scored against the cdf, not against FracF_PPF."""
    g = GAS_SN(alpha=alpha, k=k, beta=beta)
    s, mu = np.sqrt(g.var()), g.mean()
    x = mu + s * np.array([-3.0, -0.5, 0.0, 1.0, 2.5])
    assert np.max(np.abs(g._cdf_by_log_inversion(x) - np.array([g._cdf(float(v)) for v in x]))) < 1e-4
    p = np.array([0.05, 0.5, 0.9, 0.99])
    q = g._squared_ppf_by_log_inversion(p)
    lib = np.array([g._cdf(float(np.sqrt(v))) - g._cdf(float(-np.sqrt(v))) for v in q])   # P(X^2 <= q)
    assert np.max(np.abs(lib - p)) < 1e-4


@pytest.mark.parametrize("alpha, k", [(0.1, 30.2), (0.2, 15.1)])
def test_cdf_and_squared_quantiles_are_consistent_at_low_alpha(alpha, k):
    """Fails when the variants disagree with each other where no library reference exists: the cdf
    must integrate the pdf, P(X^2 <= q) must equal F(sqrt q) - F(-sqrt q), and the X^2 quantiles must
    invert P(X^2 <= q) out to the 1e-4 tails a 9,000-point QQ plot reaches."""
    g = GAS_SN(alpha=alpha, k=k, beta=-0.12)
    e = np.exp(FracChiMean(alpha, k).log_mean())
    x0 = -3.0 * np.sqrt(g.var())
    gx, gw = np.polynomial.legendre.leggauss(20)
    edges = np.linspace(-1000.0 / e, x0, 2001)
    half = np.diff(edges)[:, None] / 2
    y = ((edges[:-1, None] + edges[1:, None]) / 2 + half * gx).ravel()
    integral = np.sum((half * gw).ravel() * np.concatenate([g._pdf_by_log_inversion(c) for c in np.array_split(y, 20)]))
    assert abs(g._cdf_by_log_inversion(x0) - integral) < 1e-12

    q = g._moment(2) * np.array([0.1, 1.0, 5.0, 30.0])
    lhs = 1.0 - g._squared_sf_by_log_inversion(q)
    rhs = g._cdf_by_log_inversion(np.sqrt(q)) - g._cdf_by_log_inversion(-np.sqrt(q))
    assert np.max(np.abs(lhs - rhs)) < 1e-12

    p = np.array([1e-4, 0.01, 0.5, 0.9, 0.999, 0.9999])
    back = 1.0 - g._squared_sf_by_log_inversion(g._squared_ppf_by_log_inversion(p))
    assert np.max(np.abs(back - p) / np.minimum(p, 1 - p)) < 1e-6
