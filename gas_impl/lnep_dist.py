"""
LNEP: the exponential power kernel with lognormal volatility, a two-shape-parameter law for returns whose peak is
flatter than any normal scale mixture allows.

The distribution. W ~ EP(alpha) with unit variance (density proportional to exp(-|w/ws|^alpha), with
ws^2 = Gamma(1/alpha)/Gamma(3/alpha)) and N ~ N(0, 1) independent:

    Z = W e^{s N} is LNEP(alpha, s),     alpha > 0, s >= 0;  loc and scale come from rv_continuous.

alpha = 2 is Clark's LNN(s) (lnn_dist.py), exactly; s = 0 is the exponential power alone; (2, 0) is the normal.
All moments are finite for every (alpha, s). Closed forms (all exact):
    E Z^{2m} = ws^{2m} Gamma((2m+1)/alpha)/Gamma(1/alpha) e^{2 m^2 s^2},   Var Z = e^{2 s^2}
    1 + gamma2/3 = (kappa_EP(alpha)/3) e^{4 s^2},     SPD = pdf(0) sd = SPD_EP(alpha) e^{3 s^2/2}

pdf and cdf are one-dimensional integrals over the log-volatility n:
    pdf(z) = int phi(n) e^{-s n} f_W(z e^{-s n}) dn,     cdf(z) = int phi(n) F_W(z e^{-s n}) dn
The distribution class evaluates them by quad, split at the integrand's peak. The standalone *_by_trapezoid
functions are vectorized and work in log space: a trapezoid rule CENTRED at the integrand's peak. The pdf peak is
closed form, as in lnn_dist with 2 replaced by alpha: with c = (|z|/ws)^alpha, the log-integrand
-n^2/2 - s n - c e^{-alpha s n} peaks at n* = w/(alpha s) - s, w = LambertW(alpha^2 s^2 c e^{alpha s^2}), with
curvature -(1 + w); the cdf peak is found by Newton from there. Both log-integrands are concave with curvature at
most -1 (for alpha >= 1), so n* +- 10 holds all but e^{-50} of the mass. The kernel puts a wall exp(-c e^{-alpha s n})
of width ~1/(alpha s) into the integrand, which Gauss-Hermite cannot resolve once alpha s is large (1e-4 at
alpha = 12, s = 0.5); the integrand is entire, so the trapezoid rule converges exponentially once its step
h = min(0.05, width/4, 0.25/(alpha s)) is below the wall width.

The plane. With v1 = log(1 + gamma2/3)/4 and v2 = 2 log(sqrt(2 pi) SPD)/3, the lognormal adds exactly (s^2, s^2):

    v1 = v1_EP(alpha) + s^2,    v2 = v2_EP(alpha) + s^2,    so    v1 - v2 = D(alpha) = v1_EP(alpha) - v2_EP(alpha).

Each alpha is a 45-degree ray from its EP point. D rises from -inf (alpha -> 0) through 0 (alpha = 2, the LNN
diagonal) to D_MAX = 0.0880 (the uniform kernel, alpha -> inf), so the LNEP region is bounded by the EP curve on the
left and by the line v2 = v1 - D_MAX below. Because the gap depends on alpha alone, the method of moments is closed
form: solve D(alpha) = v1 - v2, then s^2 = v1 - v1_EP(alpha).

LNEP_Fit fits returns, X = loc + sd e^{-s^2} Z, so sd(X) = sd:
    method='mle'      maximum likelihood over (alpha, s, loc, sd) (the default); alpha is loosely identified on short
                      samples, see profile_alpha
    method='moments'  match exkurt and SPD exactly (loc = sample mean, sd = sample sd); status 'outside' when the
                      sample shape is outside the LNEP region

Usage:
    lnep.pdf(x, alpha, s), lnep.cdf(...), lnep.rvs(alpha, s, size=n), lnep.stats(alpha, s, moments='mvsk')
    f = LNEP_Fit(returns)                # simple returns as decimals; an array or a pd.Series
    f = LNEP_Fit(returns, method='moments')
    f.alpha, f.s, f.loc, f.sd, f.status
    f.summary()                          # pd.Series: the fit, the model and sample shape, loglik
    f.profile_alpha(alphas)              # pd.DataFrame: s, loc, sd, loglik maximized at each fixed alpha
    f.pdf(xg), f.cdf(xg), f.qq()         # the fitted law
    LNEP_Fit.v12(alpha, s)               # the plane point of any (alpha, s)
"""
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import brentq, minimize
from scipy.special import gammaln, gammaincc, logsumexp
from scipy.stats import rv_continuous, gennorm, lognorm

from .gsas_fit import GSaS_Fit
from .lnn_dist import _lambertw_of_exp


_LOG_2PI = np.log(2 * np.pi)
_SPAN = 10.0              # half-width of the trapezoid window in n, in units where the curvature is at least 1
_K_MAX = 5000             # cap on the trapezoid points per side


# ---------------------------------------------
# the unit-variance exponential power kernel W

def ep_ws(alpha):
    """Scale ws of the unit-variance EP(alpha): W = ws G with G ~ gennorm(alpha); ws^2 = Gamma(1/a)/Gamma(3/a)."""
    alpha = np.asarray(alpha, dtype=float)
    return np.exp(0.5 * (gammaln(1 / alpha) - gammaln(3 / alpha)))


def _ep_log_f0(alpha):
    # log density of the unit-variance EP at 0: log(alpha / (2 ws Gamma(1/alpha)))
    return np.log(alpha / 2) - np.log(ep_ws(alpha)) - gammaln(1 / alpha)


def _ep_log_c(z, alpha):
    # log (|z|/ws)^alpha; -inf at z = 0
    with np.errstate(divide='ignore'):
        return alpha * (np.log(np.abs(z)) - np.log(ep_ws(alpha)))


def _log_gammaincc(a, x):
    """log of the regularized upper incomplete gamma Q(a, x), x >= 0; asymptotic series where Q underflows."""
    a, x = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(x, dtype=float))
    q = gammaincc(a, x)
    out = np.empty_like(q)
    ok = q > 1e-280
    out[ok] = np.log(q[ok])
    if np.any(~ok):                                       # x is large: Q ~ x^{a-1} e^{-x}/Gamma(a) sum (a-1)_k / x^k
        aa, xx = a[~ok], x[~ok]
        term, ssum = np.ones_like(xx), np.ones_like(xx)
        for k in range(1, 12):
            term = term * (aa - k) / xx
            ssum = ssum + term
        with np.errstate(invalid='ignore'):
            out[~ok] = np.where(np.isinf(xx), -np.inf, (aa - 1) * np.log(xx) - xx - gammaln(aa) + np.log(ssum))
    return out


def ep_log_pdf(z, alpha):
    """log pdf of the unit-variance EP(alpha) at z."""
    with np.errstate(over='ignore'):                      # (|z|/ws)^alpha = inf is a log density of -inf
        return _ep_log_f0(alpha) - np.exp(_ep_log_c(z, alpha))


def _ep_log_cdf_neg(z, alpha):
    # log cdf of the unit-variance EP(alpha) at z <= 0: log(Q(1/alpha, (|z|/ws)^alpha)/2)
    with np.errstate(over='ignore'):
        x = np.exp(_ep_log_c(z, alpha))
    return np.log(0.5) + _log_gammaincc(1 / np.asarray(alpha, dtype=float), x)


def ep_kurtosis(alpha):
    """Kurtosis (not excess) of EP(alpha): Gamma(5/a) Gamma(1/a) / Gamma(3/a)^2."""
    alpha = np.asarray(alpha, dtype=float)
    return np.exp(gammaln(5 / alpha) + gammaln(1 / alpha) - 2 * gammaln(3 / alpha))


def ep_spd(alpha):
    """Standardized peak density sd f(0) of EP(alpha): (a/2) Gamma(3/a)^(1/2) / Gamma(1/a)^(3/2)."""
    return np.exp(_ep_log_f0(alpha))


# ---------------------------------------------
# closed forms of LNEP(alpha, s)

def lnep_var(alpha, s):     return np.exp(2.0 * np.asarray(s, dtype=float)**2) + 0.0 * np.asarray(alpha, dtype=float)
def lnep_ex_kurt(alpha, s): return ep_kurtosis(alpha) * np.exp(4.0 * np.asarray(s, dtype=float)**2) - 3.0
def lnep_spd(alpha, s):     return ep_spd(alpha) * np.exp(1.5 * np.asarray(s, dtype=float)**2)


def ep_v12(alpha):
    """Plane point (v1, v2) of EP(alpha) alone."""
    return np.log(ep_kurtosis(alpha) / 3) / 4, 2 * np.log(np.sqrt(2 * np.pi) * ep_spd(alpha)) / 3


def lnep_v12(alpha, s):
    """Plane point (v1, v2) of LNEP(alpha, s): EP(alpha)'s plus (s^2, s^2)."""
    e1, e2 = ep_v12(alpha)
    return e1 + s**2, e2 + s**2


# ---------------------------------------------
# trapezoid pdf and cdf, peak-centred, vectorized, in log space

def _centred_trapezoid_log_integral(log_f, center, scale, alpha_s):
    """log int exp(log_f(n)) dn by the trapezoid rule on center +- _SPAN, with step min(0.05, scale/4, 0.25/(alpha s))
    per point (arrays); the batch shares one number of points, set by its smallest step."""
    h = np.minimum(np.minimum(0.05, scale / 4), 0.25 / np.maximum(alpha_s, 1e-12))
    K = int(min(_K_MAX, np.ceil(_SPAN / np.min(h)))) if h.size else 1
    n = center[..., None] + h[..., None] * np.arange(-K, K + 1)
    with np.errstate(over='ignore'):                      # the far side of the wall: exp(+big) -> a log-integrand of -inf
        return np.log(h) + logsumexp(log_f(n), axis=-1)


def _lnep_pdf_peak(z, alpha, s):
    """Peak n* and curvature width of the pdf integrand -n^2/2 - s n - c e^{-alpha s n}, c = (|z|/ws)^alpha, s > 0.
    w = alpha s (n* + s) solves w e^w = alpha^2 s^2 c e^{alpha s^2}, and the curvature at n* is -(1 + w)."""
    z, alpha, s = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (z, alpha, s)))
    nz = z != 0
    L = 2 * np.log(alpha * s) + _ep_log_c(np.where(nz, z, 1.0), alpha) + alpha * s * s
    w = np.where(nz, _lambertw_of_exp(np.where(nz, L, 0.0)), 0.0)
    return w / (alpha * s) - s, 1.0 / np.sqrt(1.0 + w)


def lnep_log_pdf_by_trapezoid(z, alpha, s):
    """log pdf of the standard LNEP(alpha, s) at z by the peak-centred trapezoid rule (arrays broadcast)."""
    z, alpha, s = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (z, alpha, s)))
    pos = s > 0
    sp = np.where(pos, s, 1.0)                                       # placeholder where s = 0, replaced below
    center, scale = _lnep_pdf_peak(z, alpha, sp)
    log_c, log_f0 = _ep_log_c(z, alpha), _ep_log_f0(alpha)
    log_f = lambda n: (-0.5 * n**2 - 0.5 * _LOG_2PI - sp[..., None] * n + log_f0[..., None]
                       - np.exp(log_c[..., None] - alpha[..., None] * sp[..., None] * n))
    out = _centred_trapezoid_log_integral(log_f, center, scale, alpha * sp)
    return np.where(pos, out, ep_log_pdf(z, alpha))


def lnep_pdf_by_trapezoid(z, alpha, s):
    return np.exp(lnep_log_pdf_by_trapezoid(z, alpha, s))


def _lnep_log_cdf_neg(z, alpha, s):
    """log cdf at z <= 0, s > 0: log int phi(n) F_W(z e^{-s n}) dn, centred by damped Newton from the pdf peak."""
    z, alpha, s = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (z, alpha, s)))
    ws, log_f0 = ep_ws(alpha), _ep_log_f0(alpha)

    def parts(n):
        t = z * np.exp(-s * n)
        with np.errstate(over='ignore'):
            q = np.exp(_ep_log_c(t, alpha))                          # (|t|/ws)^alpha
        r = np.exp(log_f0 - q - _ep_log_cdf_neg(t, alpha))           # f_W(t)/F_W(t)
        G1 = -n - s * t * r                                          # d/dn [log phi(n) + log F_W(t)]
        with np.errstate(invalid='ignore'):
            G2 = -1.0 + s * s * t * r * (1.0 - alpha * q - t * r)
        G2 = np.where(np.isfinite(G2), G2, -1.0)                     # q = inf, r = 0: far past the wall
        return G1, np.minimum(G2, -1e-3)
    n, _ = _lnep_pdf_peak(z, alpha, s)
    for _ in range(60):
        G1, G2 = parts(n)
        n = n + np.clip(-G1 / G2, -2.0, 2.0)
    _, G2 = parts(n)
    scale = np.minimum(1.0 / np.sqrt(-G2), 1.0)
    log_f = lambda m: (-0.5 * m**2 - 0.5 * _LOG_2PI) + _ep_log_cdf_neg(z[..., None] * np.exp(-s[..., None] * m), alpha[..., None])
    return _centred_trapezoid_log_integral(log_f, n, scale, alpha * s)


def lnep_log_cdf_by_trapezoid(z, alpha, s):
    """log cdf of the standard LNEP(alpha, s) at z by the peak-centred trapezoid rule (arrays broadcast). The left
    tail is computed directly, the right by symmetry, so both keep relative accuracy."""
    z, alpha, s = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (z, alpha, s)))
    pos = s > 0
    zn = -np.abs(z)
    out = np.where(pos, _lnep_log_cdf_neg(zn, alpha, np.where(pos, s, 1.0)), _ep_log_cdf_neg(zn, alpha))
    return np.where(z > 0, np.log1p(-np.exp(out)), out)              # cdf(z) = 1 - cdf(-z) by symmetry


def lnep_cdf_by_trapezoid(z, alpha, s):
    return np.exp(lnep_log_cdf_by_trapezoid(z, alpha, s))


# ---------------------------------------------
# quad pdf and cdf (scalars), for the distribution class

def _lnep_pdf_by_quad(z, alpha, s):
    # Z = W v with volatility v = e^{s N} ~ lognorm(s): pdf(z) = int_0^inf (1/v) f_W(z/v) lognorm(s).pdf(v) dv,
    # split at the integrand's peak v* = e^{s n*} so quad finds the mass in the tails
    if s == 0:
        return float(np.exp(ep_log_pdf(z, alpha)))
    c, _ = _lnep_pdf_peak(z, alpha, s)
    v_peak = float(np.exp(s * c))
    vol = lognorm(s)
    f = lambda v: float(np.exp(ep_log_pdf(z / v, alpha))) / v * vol.pdf(v)
    return quad(f, 0.0, v_peak, epsabs=0, epsrel=1e-12, limit=200)[0] \
        + quad(f, v_peak, np.inf, epsabs=0, epsrel=1e-12, limit=200)[0]


def _lnep_cdf_by_quad(z, alpha, s):
    # cdf(z) = int_0^inf F_W(z/v) lognorm(s).pdf(v) dv for z <= 0 (right side by symmetry)
    if z > 0:
        return 1.0 - _lnep_cdf_by_quad(-z, alpha, s)
    if s == 0:
        return float(np.exp(_ep_log_cdf_neg(z, alpha)))
    c, _ = _lnep_pdf_peak(z, alpha, s)
    v_peak = float(np.exp(s * c))
    vol = lognorm(s)
    f = lambda v: float(np.exp(_ep_log_cdf_neg(z / v, alpha))) * vol.pdf(v)
    return quad(f, 0.0, v_peak, epsabs=0, epsrel=1e-12, limit=200)[0] \
        + quad(f, v_peak, np.inf, epsabs=0, epsrel=1e-12, limit=200)[0]


class lnep_gen(rv_continuous):
    """Exponential power with lognormal volatility, standard form: Z = W exp(s N), W ~ EP(alpha) with unit variance,
    N ~ N(0,1) independent; shapes alpha > 0, s >= 0. alpha = 2 is lnn(s). Use loc/scale for the location-scale family."""

    def _argcheck(self, alpha, s):
        return (alpha > 0) & (s >= 0)

    def _pdf(self, x, alpha, s):
        return np.vectorize(_lnep_pdf_by_quad, otypes=[float])(x, alpha, s)

    def _cdf(self, x, alpha, s):
        return np.vectorize(_lnep_cdf_by_quad, otypes=[float])(x, alpha, s)

    def _sf(self, x, alpha, s):
        return np.vectorize(_lnep_cdf_by_quad, otypes=[float])(-np.asarray(x, dtype=float), alpha, s)   # symmetric

    def _rvs(self, alpha, s, size=None,
             random_state: Optional[Union[np.random.Generator, np.random.RandomState]] = None):
        rng = np.random.default_rng() if random_state is None else random_state   # scipy always passes one
        w = ep_ws(alpha) * gennorm.rvs(alpha, size=size, random_state=rng)
        return w * np.exp(s * rng.standard_normal(size))

    def _munp(self, n, alpha, s):
        # odd moments vanish; E Z^n = ws^n Gamma((n+1)/alpha)/Gamma(1/alpha) e^{n^2 s^2 / 2} for even n
        if n % 2 == 1:
            return 0.0
        return np.exp(n * np.log(ep_ws(alpha)) + gammaln((n + 1) / alpha) - gammaln(1 / alpha) + 0.5 * n * n * s * s)

    def _stats(self, alpha, s):
        return 0.0, lnep_var(alpha, s), 0.0, lnep_ex_kurt(alpha, s)


lnep = lnep_gen(name="lnep", shapes="alpha, s")


# ---------------------------------------------
# fitting returns

class LNEP_Fit:
    SQ2PI = np.sqrt(2*np.pi)
    D_MAX = 0.0880                                   # sup of D(alpha), the uniform kernel; refined at import below
    ALPHA_BOUNDS, S_BOUNDS = (0.5, 50.0), (0.0, 1.5)

    def __init__(self, x, method='mle'):
        """x: simple returns (an array or a pd.Series, no NaN). method: 'mle' (default) or 'moments'."""
        x = np.asarray(x, dtype=float)
        if np.isnan(x).any() or len(x) < 30:
            raise ValueError("x must have no NaN and at least 30 values")
        self.x = x
        self.method = method
        self.shape = GSaS_Fit.sample_shape(x)        # dict: n, exkurt, SPD, v1, v2
        fit = {'mle': self._fit_mle, 'moments': self._fit_moments}[method]()
        self.alpha, self.s, self.loc, self.sd, self.status = fit

    # ----------------------------------------------------------------------------------------------------
    # closed forms (the module functions, as class methods)
    # ----------------------------------------------------------------------------------------------------
    ep_kurtosis = staticmethod(ep_kurtosis)
    ep_spd = staticmethod(ep_spd)
    ep_v12 = staticmethod(ep_v12)
    v12 = staticmethod(lnep_v12)
    exkurt = staticmethod(lnep_ex_kurt)
    spd = staticmethod(lnep_spd)

    @staticmethod
    def gap(alpha):
        """D(alpha) = v1_EP - v2_EP, the plane gap v1 - v2 of every LNEP(alpha, s)."""
        e1, e2 = ep_v12(alpha)
        return e1 - e2

    @classmethod
    def invert(cls, v1, v2):
        """
        Plane point -> (alpha, s) by the closed-form moments. Returns a tuple (alpha, s, status): status 'ok', or
        'outside' (alpha, s nan) when v1 - v2 >= D_MAX or the implied s^2 < 0.
        """
        g = v1 - v2
        lo, hi = 0.05, 1e4                       # D increases with alpha: very negative at 0.05, ~D_MAX at 1e4
        if g >= cls.gap(hi) or g <= cls.gap(lo):
            return np.nan, np.nan, 'outside'
        alpha = brentq(lambda a: cls.gap(a) - g, lo, hi, xtol=1e-12)
        s2 = v1 - ep_v12(alpha)[0]
        if s2 < 0:
            return np.nan, np.nan, 'outside'
        return alpha, np.sqrt(s2), 'ok'

    # ----------------------------------------------------------------------------------------------------
    # the law of X = loc + sd e^{-s^2} Z, Z ~ lnep(alpha, s)
    # ----------------------------------------------------------------------------------------------------
    @staticmethod
    def logpdf_of(xg, alpha, s, loc, sd):
        """log density of LNEP(alpha, s) with location loc and standard deviation sd at xg. Returns an ndarray."""
        c = sd * np.exp(-s**2)
        return lnep_log_pdf_by_trapezoid((np.asarray(xg, dtype=float) - loc) / c, alpha, s) - np.log(c)

    @classmethod
    def pdf_of(cls, xg, alpha, s, loc, sd):
        """The LNEP(alpha, s) density with location loc and standard deviation sd at xg. Returns an ndarray."""
        return np.exp(cls.logpdf_of(xg, alpha, s, loc, sd))

    @staticmethod
    def cdf_of(xg, alpha, s, loc, sd):
        """The LNEP(alpha, s) cdf with location loc and standard deviation sd at xg. Returns an ndarray."""
        c = sd * np.exp(-s**2)
        return lnep_cdf_by_trapezoid((np.asarray(xg, dtype=float) - loc) / c, alpha, s)

    @staticmethod
    def rvs_of(n, alpha, s, loc, sd, rng=None):
        """n draws of LNEP(alpha, s, loc, sd). rng: a np.random.Generator or a seed. Returns an ndarray."""
        return lnep.rvs(alpha, s, loc=loc, scale=sd * np.exp(-s**2), size=n, random_state=np.random.default_rng(rng))

    def pdf(self, xg):
        """The fitted density at xg (an array). Returns an ndarray of len(xg)."""
        self._require_fit()
        return self.pdf_of(xg, self.alpha, self.s, self.loc, self.sd)

    def cdf(self, xg):
        """The fitted cdf at xg (an array). Returns an ndarray of len(xg)."""
        self._require_fit()
        return self.cdf_of(xg, self.alpha, self.s, self.loc, self.sd)

    def loglik(self):
        """Log-likelihood of the fitted law on x."""
        self._require_fit()
        return float(self.logpdf_of(self.x, self.alpha, self.s, self.loc, self.sd).sum())

    def qq(self, n_grid=4001):
        """Q-Q pairs at the plotting positions (i - 1/2)/n. Returns a tuple (q_fit, x_sorted) of two ndarrays."""
        xs = np.sort(self.x)
        p = (np.arange(1, len(xs) + 1) - 0.5)/len(xs)
        span = xs[-1] - xs[0]
        grid = np.linspace(xs[0] - 0.5*span, xs[-1] + 0.5*span, n_grid)
        return np.interp(p, self.cdf(grid), grid), xs

    # ----------------------------------------------------------------------------------------------------
    # fits
    # ----------------------------------------------------------------------------------------------------
    def _require_fit(self):
        if self.status != 'ok':
            raise ValueError(f"no LNEP fit: {self.status}")

    def _fit_moments(self):
        alpha, s, status = self.invert(self.shape['v1'], self.shape['v2'])
        return alpha, s, float(self.x.mean()), float(self.x.std()), status

    def _nll(self, p, alpha=None):
        # p = (alpha, s, loc/sd_x, log(sd/sd_x)), or (s, loc/sd_x, log(sd/sd_x)) when alpha is fixed
        a, (s, loc, lsd) = (p[0], p[1:]) if alpha is None else (alpha, p)
        sx = self.x.std()
        v = -self.logpdf_of(self.x, a, s, loc*sx, np.exp(lsd)*sx).sum()
        return v if np.isfinite(v) else 1e10

    def _mle(self, alpha=None):
        sx = self.x.std()
        mu = self.x.mean()/sx
        bounds = ([] if alpha is not None else [self.ALPHA_BOUNDS]) + [self.S_BOUNDS, (mu - 1, mu + 1), (-2, 2)]
        starts = [[a, s, mu, 0.0] for a, s in [(2, 0.2), (3, 0.3), (4, 0.35), (6, 0.4)]]
        if alpha is not None:
            starts = [st[1:] for st in starts]
        best = None
        for st in starts:
            r = minimize(self._nll, st, args=(alpha,), method='L-BFGS-B', bounds=bounds)
            if best is None or r.fun < best.fun:
                best = r
        return best

    def _fit_mle(self):
        r = self._mle()
        a, s, loc, lsd = r.x
        sx = self.x.std()
        return float(a), float(s), float(loc*sx), float(np.exp(lsd)*sx), 'ok' if r.success else 'not converged'

    def profile_alpha(self, alphas):
        """
        The profile log-likelihood: at each fixed alpha, loglik maximized over (s, loc, sd). Returns a pd.DataFrame
        indexed by alpha with columns s, loc, sd (the fit at that alpha; LNEP_Fit.pdf_of/cdf_of take them), loglik,
        drop (loglik below the profile maximum; drop < 1.92 is the 95% likelihood-ratio interval for alpha), v1, v2
        (the model's plane point).
        """
        sx = self.x.std()
        rows = []
        for a in alphas:
            r = self._mle(alpha=a)
            s, loc, lsd = (float(v) for v in r.x)
            v1, v2 = lnep_v12(a, s)
            rows.append(dict(alpha=a, s=s, loc=loc*sx, sd=np.exp(lsd)*sx, loglik=-r.fun, v1=v1, v2=v2))
        out = pd.DataFrame(rows).set_index('alpha')
        out.insert(4, 'drop', out.loglik.max() - out.loglik)
        return out

    def summary(self):
        """pd.Series: method, status, alpha, s, loc, sd, the model's exkurt, SPD, v1, v2, the sample's, and loglik."""
        ok = self.status == 'ok'
        v1, v2 = lnep_v12(self.alpha, self.s) if ok else (np.nan, np.nan)
        return pd.Series(dict(method=self.method, status=self.status, alpha=self.alpha, s=self.s, loc=self.loc, sd=self.sd,
                              exkurt=lnep_ex_kurt(self.alpha, self.s) if ok else np.nan,
                              SPD=lnep_spd(self.alpha, self.s) if ok else np.nan, v1=v1, v2=v2,
                              sample_exkurt=self.shape['exkurt'], sample_SPD=self.shape['SPD'],
                              sample_v1=self.shape['v1'], sample_v2=self.shape['v2'],
                              loglik=self.loglik() if ok else np.nan))


LNEP_Fit.D_MAX = float(LNEP_Fit.gap(1e6))
