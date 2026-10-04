
# Clark (1973) LNN: Lognormal-normal distribution
# X ~ N(0, s^2), Y ~ N(0,1)
# Z ~ Y exp(X) is LNN(s)
#
# Also known as the normal log-normal (NLN) mixture (Tauchen & Pitts 1983, Hsieh 1989, Yang 2008),
# where NLN usually allows X and Y to be correlated; LNN here is the uncorrelated, symmetric case,
# Clark's Theorem 5. s > 0 is the only shape parameter (location/scale come from rv_continuous).
#
# Closed forms (all exact):
#     E Z^{2m} = (2m-1)!! exp(2 m^2 s^2),   Var Z = exp(2 s^2),   1 + gamma2/3 = exp(4 s^2)
#     pdf(0) = exp(s^2/2) / sqrt(2 pi),     SPD = pdf(0) * sd = exp(3 s^2 / 2) / sqrt(2 pi)
# so 1 + gamma2/3 = (sqrt(2 pi) SPD)^(8/3): the large-|k| terminal curve of GSaS.
#
# pdf and cdf are one-dimensional integrals over the log-volatility n, X = s n:
#     pdf(z) = int phi(n) e^{-s n} phi(z e^{-s n}) dn,     cdf(z) = int phi(n) Phi(z e^{-s n}) dn
# The distribution class evaluates them by quad. The standalone *_by_gauss_hermite functions use
# Gauss-Hermite CENTRED at the integrand's peak and scaled by its curvature, so the rule follows the
# mass into the tails (a fixed rule around n = 0 misses it there); they are vectorized and work in
# log space.

import numpy as np
from typing import Optional, Union
from scipy.stats import rv_continuous, norm, lognorm
from scipy.integrate import quad
from scipy.special import lambertw, log_ndtr, logsumexp, factorial2


# Gauss-Hermite nodes/weights for int e^{-x^2} f(x) dx. The integrands get less Gaussian as s grows;
# at s = 1.5 the pdf needs 128 nodes (64 gave 1e-10) and the cdf 200 (128 gave 1e-12) for ~1e-15.
_GH_PDF = np.polynomial.hermite.hermgauss(128)
_GH_CDF = np.polynomial.hermite.hermgauss(200)
_LOG_2PI = np.log(2 * np.pi)


def _lambertw_of_exp(L):
    """W(e^L) for real L, without forming e^L (it overflows past L ~ 709). W solves w + log w = L."""
    L = np.asarray(L, dtype=float)
    w = np.empty_like(L)
    small = L < 700
    w[small] = np.real(lambertw(np.exp(L[small])))
    big = ~small
    if np.any(big):
        wb = L[big] - np.log(L[big])
        for _ in range(50):                               # Newton on w + log w - L
            wb = wb - (wb + np.log(wb) - L[big]) / (1 + 1 / wb)
        w[big] = wb
    return w


def _centred_gh_log_integral(log_f, center, scale, rule):
    """log int exp(log_f(n)) dn by Gauss-Hermite `rule` = (nodes, weights), centred at `center` with
    width `scale` (arrays)."""
    gh_x, gh_w = rule
    n = center[..., None] + np.sqrt(2.0) * scale[..., None] * gh_x
    vals = log_f(n) + gh_x**2 + np.log(gh_w)
    return np.log(np.sqrt(2.0) * scale) + logsumexp(vals, axis=-1)


def _lnn_pdf_peak(z, s):
    """Peak n* and curvature width of the pdf integrand g(n) = -n^2/2 - s n - z^2 e^{-2 s n}/2.
    g'(n) = 0 gives n + s = s z^2 e^{-2 s n}, i.e. w e^w = 2 s^2 z^2 e^{2 s^2} with w = 2 s (n + s),
    so n* = w/(2s) - s with w = LambertW(.) in closed form, and g''(n*) = -(1 + w)."""
    z, s = np.broadcast_arrays(np.asarray(z, dtype=float), np.asarray(s, dtype=float))
    nz = z != 0
    with np.errstate(divide='ignore'):
        L = np.log(2 * s * s) + 2 * np.log(np.abs(z)) + 2 * s * s    # never form z^2: overflows past 1e154
    w = np.where(nz, _lambertw_of_exp(np.where(nz, L, 0.0)), 0.0)
    return w / (2 * s) - s, 1.0 / np.sqrt(1.0 + w)


def lnn_log_pdf_by_gauss_hermite(z, s):
    """log pdf of the standard LNN(s) at z by peak-centred Gauss-Hermite (arrays broadcast).
    Vectorized, and accurate in the far tails because the rule follows the integrand's peak."""
    z, s = np.broadcast_arrays(np.asarray(z, dtype=float), np.asarray(s, dtype=float))
    center, scale = _lnn_pdf_peak(z, s)
    with np.errstate(divide='ignore'):
        log_abs_z = np.log(np.abs(z))                                # z^2 e^{-2sn} = exp(2 log|z| - 2sn)
    log_f = lambda n: -0.5 * n**2 - s[..., None] * n - 0.5 * np.exp(2 * log_abs_z[..., None] - 2 * s[..., None] * n)
    return _centred_gh_log_integral(log_f, center, scale, _GH_PDF) - _LOG_2PI


def lnn_pdf_by_gauss_hermite(z, s):
    return np.exp(lnn_log_pdf_by_gauss_hermite(z, s))


def _lnn_log_cdf_neg(z, s):
    """log cdf at z <= 0: log int phi(n) Phi(z e^{-s n}) dn. The log-integrand is concave in n
    (log Phi is concave and increasing, z e^{-s n} is concave for z <= 0), so Newton finds its peak."""
    z, s = np.broadcast_arrays(np.asarray(z, dtype=float), np.asarray(s, dtype=float))
    s_ = s
    def parts(n):
        t = z * np.exp(-s_ * n)
        log_phi_t = -0.5 * t * t - 0.5 * _LOG_2PI
        r = np.exp(log_phi_t - log_ndtr(t))                # phi(t)/Phi(t)
        G1 = -n - s_ * t * r                               # d/dn [log phi(n) + log Phi(t)]
        G2 = -1.0 + s_ * s_ * t * r * (1.0 - t * t - t * r)
        return G1, G2
    # start from the pdf's peak, then Newton (damped) on the concave log-integrand
    n, _ = _lnn_pdf_peak(z, s)
    for _ in range(60):
        G1, G2 = parts(n)
        step = -G1 / G2
        n = n + np.clip(step, -2.0, 2.0)
    _, G2 = parts(n)
    scale = 1.0 / np.sqrt(-G2)
    log_f = lambda m: (-0.5 * m**2 - 0.5 * _LOG_2PI) + log_ndtr(z[..., None] * np.exp(-s_[..., None] * m))
    return _centred_gh_log_integral(log_f, n, scale, _GH_CDF)


def lnn_log_cdf_by_gauss_hermite(z, s):
    """log cdf of the standard LNN(s) at z by peak-centred Gauss-Hermite (arrays broadcast). The left
    tail is computed directly, the right by symmetry, so both keep relative accuracy."""
    z, s = np.broadcast_arrays(np.asarray(z, dtype=float), np.asarray(s, dtype=float))
    out = _lnn_log_cdf_neg(-np.abs(z), s)                  # log cdf(-|z|), accurate in the left tail
    pos = z > 0
    out = np.where(pos, np.log1p(-np.exp(out)), out)       # cdf(z) = 1 - cdf(-z) by symmetry
    return out


def lnn_cdf_by_gauss_hermite(z, s):
    return np.exp(lnn_log_cdf_by_gauss_hermite(z, s))


def _lnn_pdf_by_quad(z, s):
    # scalar: Z = Y v with volatility v = e^{s N} ~ lognorm(s), so
    #     pdf(z) = int_0^inf (1/v) phi(z/v) lognorm(s).pdf(v) dv
    # split at the integrand's peak v* = e^{s n*} so quad finds the mass in the tails
    c, _ = _lnn_pdf_peak(z, s)
    v_peak = float(np.exp(s * c))
    vol = lognorm(s)
    f = lambda v: norm.pdf(z / v) / v * vol.pdf(v)
    return quad(f, 0.0, v_peak, epsabs=0, epsrel=1e-12, limit=200)[0] \
        + quad(f, v_peak, np.inf, epsabs=0, epsrel=1e-12, limit=200)[0]


def _lnn_cdf_by_quad(z, s):
    # scalar: cdf(z) = int_0^inf Phi(z/v) lognorm(s).pdf(v) dv for z <= 0 (right side by symmetry)
    if z > 0:
        return 1.0 - _lnn_cdf_by_quad(-z, s)
    c, _ = _lnn_pdf_peak(z, s)
    v_peak = float(np.exp(s * c))
    vol = lognorm(s)
    f = lambda v: norm.cdf(z / v) * vol.pdf(v)
    return quad(f, 0.0, v_peak, epsabs=0, epsrel=1e-12, limit=200)[0] \
        + quad(f, v_peak, np.inf, epsabs=0, epsrel=1e-12, limit=200)[0]


def lnn_var(s):         return np.exp(2.0 * np.asarray(s)**2)
def lnn_ex_kurt(s):     return 3.0 * np.exp(4.0 * np.asarray(s)**2) - 3.0
def lnn_spd(s):         return np.exp(1.5 * np.asarray(s)**2) / np.sqrt(2 * np.pi)
def lnn_s_from_ex_kurt(gamma2):
    """Shape s of the LNN with excess kurtosis gamma2 (> 0): 1 + gamma2/3 = exp(4 s^2)."""
    return np.sqrt(np.log1p(np.asarray(gamma2) / 3.0) / 4.0)


class lnn_gen(rv_continuous):
    """Clark (1973) lognormal-normal distribution, standard form: Z = Y exp(s N), Y, N ~ N(0,1)
    independent, shape s > 0. Use loc/scale for the location-scale family."""

    def _argcheck(self, s):
        return s > 0

    def _pdf(self, x, s):
        return np.vectorize(_lnn_pdf_by_quad, otypes=[float])(x, s)

    def _cdf(self, x, s):
        return np.vectorize(_lnn_cdf_by_quad, otypes=[float])(x, s)

    def _sf(self, x, s):
        return np.vectorize(_lnn_cdf_by_quad, otypes=[float])(-np.asarray(x, dtype=float), s)   # symmetric

    def _rvs(self, s, size=None,
             random_state: Optional[Union[np.random.Generator, np.random.RandomState]] = None):
        rng = np.random.default_rng() if random_state is None else random_state   # scipy always passes one
        y = rng.standard_normal(size)
        n = rng.standard_normal(size)
        return y * np.exp(s * n)

    def _munp(self, n, s):
        # odd moments vanish; E Z^{2m} = (2m-1)!! exp(2 m^2 s^2)
        if n % 2 == 1:
            return 0.0
        m = n // 2
        return factorial2(2 * m - 1, exact=False) * np.exp(2.0 * m * m * s * s) if m > 0 else 1.0

    def _stats(self, s):
        return 0.0, lnn_var(s), 0.0, lnn_ex_kurt(s)


lnn = lnn_gen(name="lnn", shapes="s")


# ---------------------------------------------
# relation between SPD and kurtosis
# 1 + ex_kurt / 3 = ( sqrt(2 pi) * SPD )**(8/3)

def lnn_from_spd_to_ex_kurt(spd):
    """Excess kurtosis on the LNN curve at standardized peak density `spd`:
    ex_kurt = 3 ((sqrt(2 pi) SPD)^(8/3) - 1), written with expm1 so it stays accurate near the Gaussian
    point (spd -> 1/sqrt(2 pi), ex_kurt -> 0). An spd below 1/sqrt(2 pi) gives a negative value,
    outside the LNN family (every LNN has ex_kurt > 0)."""
    return 3.0 * np.expm1((8.0 / 3.0) * np.log(np.sqrt(2 * np.pi) * np.asarray(spd, dtype=float)))


def lnn_from_ex_kurt_to_spd(ex_kurt):
    """Standardized peak density on the LNN curve at excess kurtosis `ex_kurt` (> -3):
    SPD = (1 + ex_kurt/3)^(3/8) / sqrt(2 pi), the inverse of from_spd_to_ex_kurt."""
    return np.exp(0.375 * np.log1p(np.asarray(ex_kurt, dtype=float) / 3.0)) / np.sqrt(2 * np.pi)
