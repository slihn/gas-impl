
import numpy as np 
import pandas as pd
import mpmath as mp
from typing import Union, Optional
from functools import lru_cache, cached_property
from scipy.stats import rv_continuous, multivariate_normal
from scipy.special import gamma, iv, loggamma, digamma, polygamma
from scipy.integrate import quad


from .wright import wright_fn, mp_gamma, levy_stable_extremal
from .wright_levy_asymp import wright_f_fn_by_levy_asymp
from .mellin import pdf_by_mellin
from .frac_gamma_dist import frac_gamma, frac_gamma_star, fg_q_by_f
from .utils import calc_stats_from_moments


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# FCM
def fcm_sigma(alpha: float, k: float, theta: float = 0.0):
    # the sign of k is irrelevant
    g = (alpha - theta) / (2.0 * alpha)  # instead of 0.5
    return abs(k)**(g - 1/alpha) * (g**g)


def fcm_moment(n: float, alpha: float, k: float, theta: float = 0.0, k_mean=True):
    # k_mean = False is for classic chi distribution
    # we use mpf for cancellation and large k, not for output, but you need to set the precision
    n = mp.mpf(n)
    k = mp.mpf(k)
    alpha = mp.mpf(alpha)
    g = (alpha - theta) / (mp.mpf(2) * alpha)  # instead of 0.5
    assert k != 0
    sigma = mp.power(abs(k), g - 1/alpha) * mp.power(g, g)
    if k > 0:
        sigma_n = mp.power(sigma, n) if k_mean else mp.power(g, n/2.0)
        c = mp_gamma((k-1)*g) / mp_gamma((k-1)/alpha) if k != 1.0 else 1/(alpha*g)
        d = mp_gamma((k+n-1)/alpha) / mp_gamma((k+n-1)*g) if k+n != 1.0 else alpha*g
        return float(sigma_n * c * d)
    if k < 0:
        assert k_mean == True
        k = abs(k)
        sigma_n = mp.power(sigma, -n)
        c = mp_gamma(k*g) / mp_gamma(k/alpha)
        d = mp_gamma((k-n)/alpha) / mp_gamma((k-n)*g) if n != k else alpha*g
        return float(sigma_n * c * d)
    raise Exception(f"ERROR: k={k} is not handled properly")


def fcm_normalization_constant(alpha: float, k: float, theta: float = 0.0):
    # this uses mp for testing purpose
    k = mp.mpf(k)
    alpha = mp.mpf(alpha)
    g = (alpha - theta) / (2.0 * alpha)  # instead of 0.5
    c = alpha * mp_gamma((k-1)*g) / mp_gamma((k-1)/alpha) if k != 1.0 else 1/g
    return float(c)


def fcm_mellin_transform(s, eps: float, k: float, g: float):
    # this uses (eplison, gamma) representation
    assert k != 0
    sigma = g**g * abs(k)**(g - eps)
    if k > 0:
        gamma_k = eps / g if k == 1.0 else gamma(g * (k-1.0)) / gamma(eps * (k-1.0))
        sigma_const = sigma**(s-1.0)
        gamma_ks = g / eps if s == 2.0-k else gamma(eps * (s+k-2.0)) / gamma(g * (s+k-2.0)) 
        return sigma_const * gamma_k * gamma_ks 
    if k < 0:
        k = abs(k)
        gamma_k = gamma(g * k) / gamma(eps * k)
        sigma_const = sigma**(1.0-s)
        gamma_ks = gamma(eps * (k-s+1.0)) / gamma(g * (k-s+1.0)) 
        return sigma_const * gamma_k * gamma_ks 
    raise Exception(f"ERROR: k={k} is not handled properly")


def fcm_k1_mellin_transform(s, eps: float, g: float):
    # this uses (eplison, gamma) representation
    gamma_s = gamma(eps * (s-1.0)) / gamma(g * (s-1.0)) 
    return eps * g**((s-1)*g - 1.0) * gamma_s 


# ---- cancellation-free lnGamma(x + h) - lnGamma(x), for the log-space inversion of FracChiMean ----
# At large |k| the two log-gammas are ~1e8 or more while their difference is O(1); subtracting them in
# double precision loses ~1e-8. These helpers keep the difference at machine precision.
_STIRLING_C = (1/12, -1/360, 1/1260, -1/1680, 1/1188, -691/360360, 1/156, -3617/122400)


def _clog1p(w):
    # accurate complex log(1 + w) for tiny |w| (numpy's complex log1p is not)
    a, b = np.real(w), np.imag(w)
    return 0.5 * np.log1p(2*a + a*a + b*b) + 1j * np.arctan2(b, 1.0 + a)


def _cexpm1(w):
    # accurate complex exp(w) - 1 for tiny |w|
    a, b = np.real(w), np.imag(w)
    return (np.expm1(a) * np.cos(b) - 2.0 * np.sin(b / 2)**2) + 1j * np.exp(a) * np.sin(b)


def loggamma_diff(x: float, h, x_min: float = 20.0):
    """lnGamma(x + h) - lnGamma(x) for real x > 0 and complex h, at machine precision for any x
    (checked against mpmath to 4e-16 for x in [20, 1e12], |h| up to 150). Below x_min the direct
    difference has no cancellation; above it, the Stirling series written as a difference:

        (x - 1/2) log1p(h/x) + h log(x + h) - h + sum_n c_n x^(1-2n) expm1((1-2n) log1p(h/x))
    """
    x = float(x)
    h = np.asarray(h, dtype=complex)          # scalar or array
    if x < x_min:
        return loggamma(x + h) - loggamma(x)
    r = _clog1p(h / x)
    out = (x - 0.5) * r + h * (np.log(x) + r) - h
    for n, c in enumerate(_STIRLING_C, start=1):
        m = 2 * n - 1
        out += c * x**(-m) * _cexpm1(-m * r)
    return out


class FracChiMean:
    # this is primarily to facilitate the testings, especially for the Mellin transform
    def __init__(self, alpha, k, theta=0.0):
        self.alpha = alpha
        self.k = k
        self.theta = theta

        self.eps = 1.0 / alpha
        self.g = (alpha - theta) / (2.0 * alpha)

        self.c = 2.0 - k + 0.5 if k > 0 else abs(k) + 1.0 - 0.5  # for Mellin inverse integration

    @cached_property
    def fcm(self):
        # built on first use: frac_chi_mean asserts sigma > 0, which underflows at large |k|, and
        # pdf_by_log_inversion never needs it
        return frac_chi_mean(self.alpha, self.k, self.theta)

    def pdf(self, x): 
        return self.fcm.pdf(x)  # type: ignore
   
    def pdf_by_mellin(self, x):
        assert x > 0
        return pdf_by_mellin(x, lambda s: fcm_mellin_transform(s, self.eps, self.k, self.g), c=self.c)

    def pdf_by_wright_f(self, x):
        assert x > 0
        C = fcm_normalization_constant(self.alpha, self.k, self.theta)
        sigma = fcm_sigma(self.alpha, self.k, self.theta)
        z = x / sigma
        return C/sigma * z**(self.k-2) * wright_f_fn_by_levy_asymp(z**self.alpha, alpha=self.alpha * self.g)

    # ---- log-space inversion: stable at large |k| and small alpha ----------------------------------
    # With V ~ FCM and eps = 1/alpha, the Mellin moment (fcm_moment) gives, exactly,
    #     log E[V^s] = +- s log(sigma) + K(s),   K(s) = A(z +- s) - A(z),   A(x) = lnGamma(eps x) - lnGamma(g x)
    # (upper signs and z = k - 1 for k > 0; lower signs and z = |k| for k < 0). U = log V - E[log V] is O(1)
    # at every k, and its characteristic function exp(K(it) - it K'(0)) is a difference of complex
    # log-gammas, so inverting it never forms the huge/tiny factors whose cancellation breaks pdf().

    def _log_inversion_branch(self):
        if self.k > 0:
            assert self.k > 1, "pdf_by_log_inversion needs k > 1 or k < 0"
            return self.k - 1.0, 1.0
        return abs(self.k), -1.0

    def _log_cgf(self, s):
        # K(s) = log E[V^s] minus the +-s log(sigma) term; s may be complex
        # A(z + sg s) - A(z), A(x) = lnGamma(eps x) - lnGamma(g x), with no cancellation
        z, sg = self._log_inversion_branch()
        return (loggamma_diff(self.eps * z, self.eps * sg * s)
                - loggamma_diff(self.g * z, self.g * sg * s))

    def log_mean(self) -> float:
        # E[log V] = +- log(sigma) + K'(0), exact. log(sigma) analytically: sigma = |k|^(g - eps) g^g
        # underflows to 0 at large |k| (e.g. 1000^-249.5), so never take np.log(fcm_sigma(...)) here
        z, sg = self._log_inversion_branch()
        log_sigma = (self.g - self.eps) * np.log(abs(self.k)) + self.g * np.log(self.g)
        k1 = sg * (self.eps * digamma(z * self.eps) - self.g * digamma(self.g * z))
        return float(sg * log_sigma + k1)

    def log_var(self) -> float:
        # Var[log V] = K''(0), exact
        z, _ = self._log_inversion_branch()
        return float(self.eps**2 * polygamma(1, z * self.eps) - self.g**2 * polygamma(1, self.g * z))

    def _phi_centered(self, t):
        # characteristic function of U = log V - E[log V] at real t (scalar or array)
        z, sg = self._log_inversion_branch()
        k1 = sg * (self.eps * digamma(z * self.eps) - self.g * digamma(self.g * z))
        return np.exp(self._log_cgf(1j * np.asarray(t, dtype=float)) - 1j * np.asarray(t) * k1)

    def _t_max(self, t_sd: float = 40.0, phi_tol: float = 1e-17, max_t_sd: float = 1e5) -> float:
        # cutoff where |phi| < phi_tol. phi decays like a Gaussian when log V is near normal, but much
        # more slowly (exponentially in t) when log V is far from normal, e.g. alpha -> 2 with small |k|,
        # so extend from t_sd / sd until the tail is really negligible
        sd = np.sqrt(self.log_var())
        t = t_sd / sd
        while np.abs(self._phi_centered(t)) > phi_tol and t * sd < max_t_sd:
            t *= 1.5
        return float(t)

    def log_v_pdf(self, u, method: str = 'trapezoid', t_sd: float = 40.0, width_sd: float = 60.0,
                  phi_tol: float = 1e-17):
        """Density of U = log V - E[log V] at u (scalar or array), by Fourier inversion of its exact
        characteristic function in centred log space:

            f_U(u) = (1/(2 pi)) int Re[ phi(t) exp(-it u) ] dt,   phi(t) = exp(K(it) - it K'(0))

        U is O(1) at every k, so this is stable where pdf() fails (large |k|, small alpha).

        method='trapezoid' (default): the integrand is even and analytic in t, so the trapezoid rule on
        the whole line converges exponentially. Step h = 2 pi / (width_sd sd): the only error is aliasing,
        ~ f_U(u +- width_sd sd), negligible for |u| well inside width_sd/2 sd. The sum stops where
        |phi| < phi_tol (see _t_max), starting from t_sd / sd. One phi evaluation serves all u.
        method='quad': adaptive quadrature per u, a slower cross-check.

        Accuracy is absolute (~1e-13 in f_U), so relative accuracy degrades in the far tails.
        """
        sd = np.sqrt(self.log_var())
        t_max = self._t_max(t_sd=t_sd, phi_tol=phi_tol)
        u_arr = np.atleast_1d(np.asarray(u, dtype=float))
        if method == 'quad':
            out = np.array([quad(lambda t: np.real(self._phi_centered(t) * np.exp(-1j * t * ui)),
                                 0.0, t_max, limit=1000)[0] / np.pi for ui in u_arr])
        elif method == 'trapezoid':
            h = 2 * np.pi / (width_sd * sd)
            t = h * np.arange(1, int(np.ceil(t_max / h)) + 1)
            phi = self._phi_centered(t)
            out = (h / np.pi) * (0.5 + np.real(np.exp(-1j * np.outer(u_arr, t)) @ phi))   # phi(0) = 1
        else:
            raise ValueError(f"method must be 'trapezoid' or 'quad', got {method!r}")
        return out if np.ndim(u) else float(out[0])

    def log_pdf_by_log_inversion(self, log_x, t_sd: float = 40.0) -> float:
        """log f_V(x) at x = exp(log_x). Takes log x because at large |k| x itself is not a float:
        E[log V] ~ 1380 at k = 1000 on the lognormal path, beyond float's e^709."""
        return float(np.log(self.log_v_pdf(log_x - self.log_mean(), t_sd=t_sd)) - log_x)

    def pdf_by_log_inversion(self, x, log: bool = False, t_sd: float = 40.0):
        """pdf of V at x via log_v_pdf: f_V(x) = f_U(log x - E[log V]) / x. log=True returns log f_V(x).
        For x beyond float range use log_pdf_by_log_inversion(log_x)."""
        assert x > 0
        lp = self.log_pdf_by_log_inversion(np.log(x), t_sd=t_sd)
        return lp if log else float(np.exp(lp))

    def cdf_by_gamma_star(self, x):
        assert self.k > 0
        z = x / fcm_sigma(self.alpha, self.k, self.theta)
        k1 = self.k - 1.0 + self.alpha
        return z**k1 * frac_gamma_star(s=k1 / self.alpha, x=z**self.alpha, alpha=self.alpha/2)
        
    @lru_cache(maxsize=10)
    def moment(self, n) -> float:
        return fcm_moment(n, self.alpha, self.k, self.theta)
    
    @lru_cache(maxsize=2)
    def calc_stats_from_moments(self):
        moments = pd.Series({
            n: self.moment(n) for n in range(5)
        })
        return calc_stats_from_moments(moments)

    @lru_cache(maxsize=2)
    def mean(self):
        return self.calc_stats_from_moments()['mean']

    @lru_cache(maxsize=2)
    def variance(self):
        return self.calc_stats_from_moments()['var']
    
    @lru_cache(maxsize=2)
    def skewness(self):
        return self.calc_stats_from_moments()['skew']
    
    @lru_cache(maxsize=2)
    def kurtosis(self):
        # this is excess kurtosis (normal kurtosis is 0.0)
        return self.calc_stats_from_moments()['kurtosis']


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
def fcm_q_by_f(z, dz_ratio: Optional[float], alpha: float):
    assert isinstance(z, float)
    assert isinstance(alpha, float)
    z = z**alpha
    if z == 0: z = 0.001
    f = wright_f_fn_by_levy_asymp(z, alpha/2)
    assert isinstance(f, float)
    assert abs(f) > 0, f"ERROR: z = {z}, f = {f} for alpha {alpha}"
    if dz_ratio is None:
        # use ratio formula of the Wright functions directly
        q = wright_fn(-z, -alpha/2, -1.0) / -f  # type: ignore
    else:
        dz = z * dz_ratio
        f_dz =  wright_f_fn_by_levy_asymp(z+dz, alpha/2)
        assert isinstance(f_dz, float)
        q = (alpha/2 * z) * (f_dz - f)/dz/f + 1
    return q 


def fcm_q_by_fg_q(z, dz_ratio: Optional[float], alpha: float):
    # this is just for testing
    return fg_q_by_f(z**alpha, dz_ratio, alpha/2)


def fcm_mu_by_f(x, dz_ratio, alpha, k):
    assert isinstance(x, float)
    alpha = float(alpha)
    k = float(k)
    # mu(x) in the moidified CIR model
    # if dz_pct is None, the use Wright function, typically good for small x < 0.3
    # dz_ratio is typcally 0.0001
    sigma = fcm_sigma(alpha, k)
    # TODO what happen x = 0 and k < 0?

    if k > 0:
        z = x / sigma
    else:
        assert x > 0  # otherwise z = np.inf
        z = (x * sigma)**(-1)

    q = fcm_q_by_f(z, dz_ratio, alpha)
    assert isinstance(q, float)
    if k > 0:
        return q + (k - 3.0)/2.0
    else:
        return -q + (1 - abs(k)/2)


def fcm_inverse_mu_by_f(x, dz_ratio, alpha, k):
    assert isinstance(x, float)
    alpha = float(alpha)
    k = float(k)
    sigma = fcm_sigma(alpha, k)
    assert k < 0, f"ERROR: only support negative k, primarily for GEP's product simulation"
    z = x / sigma
    q = fcm_q_by_f(z, dz_ratio, alpha)
    assert isinstance(q, float)
    return q + (abs(k)/2 - 1)


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
@lru_cache(maxsize=100)
def frac_chi_mean(alpha, k, theta=0.0, loc=0.0, scale=1.0):
    alpha = float(alpha)
    k = float(k)
    assert k != 0, Exception(f"ERROR: k cannot be zero in frac_chi_mean")

    g = (alpha - theta) / (2.0 * alpha)  # instead of 0.5
    sigma = fcm_sigma(alpha, k, theta)
    assert sigma > 0

    if k > 0:  return frac_gamma(alpha=alpha*g, sigma=sigma,   d=k-1, p = alpha,  loc=loc, scale=scale)
    if k < 0:  return frac_gamma(alpha=alpha*g, sigma=1/sigma, d=k,   p = -alpha, loc=loc, scale=scale)
    raise Exception(f"ERROR: k is not handled properly")


# alias, make it simpler to use
def fcm(alpha, k, theta=0.0, loc=0.0, scale=1.0):  return frac_chi_mean(alpha, k, theta=theta, loc=loc, scale=scale)


# Note: use fcm_inverse() as the volatility process for GAS and GEP
def fcm_inverse(alpha, k, theta=0.0, loc=0.0, scale=1.0):
    g = (alpha - theta) / (2.0 * alpha)  # instead of 0.5
    sigma = fcm_sigma(alpha, k, theta)
    if k > 0: return frac_gamma(alpha=alpha*g, sigma=1/sigma, d=-(k-1), p = -alpha, loc=loc, scale=scale)  # \chi^\dagger of \chi, generates GAS
    if k < 0: return frac_gamma(alpha=alpha*g, sigma=sigma,   d=abs(k), p = alpha,  loc=loc, scale=scale)  # \chi^\dagger_\phi of \chi, generates GEP
    raise Exception(f"ERROR: k is not handled properly")


def fcm_characteristic(alpha, k, theta=0.0, loc=0.0, scale=1.0):
    assert k > 0, "ERROR: characteristic function is only defined for k > 0"    
    return frac_chi_mean(alpha, -k, theta=theta, loc=loc, scale=scale)  # \chi_\phi of \chi


def fcm_characteristic_inverse(alpha, k, theta=0.0, loc=0.0, scale=1.0):
    assert k > 0, "ERROR: characteristic function is only defined for k > 0"    
    return fcm_inverse(alpha, -k, theta=theta, loc=loc, scale=scale)  # \chi^\dagger_\phi of \chi


def fcm_characteristic_pdf(x, alpha, k, use_relation=True):
    # this is not meant to be efficient, just used for proof
    assert k > 0, "ERROR: only support positive k for fcm_characteristic_pdf"
    if use_relation:
        # use relation between characteristic and original fcm pdf
        m1 = fcm_moment(1.0, alpha, k)
        pdf = frac_chi_mean(alpha=alpha, k=k).pdf(1/x)  # type: ignore 
        return pdf * x**(-3) / m1
    else:
        # direct formula
        C = fcm_normalization_constant(alpha, k+1)
        sigma = fcm_sigma(alpha, k)
        z = x * sigma
        return C * sigma**(-k) * x**(-k-1) * wright_f_fn_by_levy_asymp(z**(-alpha), alpha=alpha/2)


def fcm_inverse_pdf(x, alpha, k, use_relation=True):
    # this is not meant to be efficient, just used for proof
    if use_relation:
        # use relation between inverse and original fcm pdf, k can be positive or negative
        m1 = fcm_moment(1.0, alpha, -k)
        pdf = frac_chi_mean(alpha=alpha, k=-k).pdf(x)  # type: ignore 
        return pdf * x / m1
    else:
        # direct formula
        assert k > 0, "ERROR: only support positive k for fcm_inverse_pdf"
        C = fcm_normalization_constant(alpha, k)
        sigma = fcm_sigma(alpha, k)
        z = x * sigma
        return C * sigma**(-k+1) * x**(-k) * wright_f_fn_by_levy_asymp(z**(-alpha), alpha=alpha/2) 


def fcm_characteristic_inverse_pdf(x, alpha, k, use_relation=True):
    # this is not meant to be efficient, just used for proof
    assert k > 0, "ERROR: only support positive k for fcm_characteristic_inverse_pdf"
    if use_relation:
        # use relation between characteristic inverse and original fcm pdf
        m1 = fcm_moment(1.0, alpha, k)
        pdf = frac_chi_mean(alpha=alpha, k=k).pdf(x)  # type: ignore 
        return pdf * x / m1
    else:
        # direct formula
        C = fcm_normalization_constant(alpha, k+1)
        sigma = fcm_sigma(alpha, k)
        z = x / sigma
        return C * sigma**(-k) * x**(k-1) * wright_f_fn_by_levy_asymp(z**(alpha), alpha=alpha/2)


def fcm_inverse_pdf_by_extremal(x, alpha, k):
    # this is not meant to be efficient, just used for proof
    assert k > 0, "ERROR: only support positive k for fcm_inverse_pdf_by_extremal"
    C = fcm_normalization_constant(alpha, k)
    sigma = fcm_sigma(alpha, k)
    z = x * sigma
    rv = levy_stable_extremal(alpha/2)
    pdf = C * sigma**(-k+3) * x**(-k+2) * rv.pdf(z**2)  # type: ignore
    return pdf

def fcm_characteristic_pdf_by_extremal(x, alpha, k):
    # this is not meant to be efficient, just used for proof
    assert k > 0, "ERROR: only support positive k for fcm_inverse_pdf_by_extremal"
    C = fcm_normalization_constant(alpha, k+1)
    sigma = fcm_sigma(alpha, k)
    z = x * sigma
    rv = levy_stable_extremal(alpha/2)
    pdf = C * sigma**(-k+2) * x**(-k+1) * rv.pdf(z**2)  # type: ignore
    return pdf


# --------------------------------------------------------------------------------
def fcm_pdf_large_x(x, alpha, k):
    # this formula doesn't work for small alpha
    alpha = float(alpha)
    k = float(k)
    assert k > 0

    # a(alpha) and b2(alpha): these are gsc's alpha between 0 and 1
    def a(alpha): return (1.0-alpha) * alpha**(alpha/(1.0-alpha))
    def b2(alpha): return alpha**(0.5/(1.0-alpha)) / np.sqrt((1.0-alpha) * 2.0*np.pi)

    pfrac = 2*alpha/(2.0-alpha)
    sigma = fcm_sigma(alpha, k)
    c = alpha / sigma * (mp_gamma((k-1)/2) / mp_gamma((k-1)/alpha) if k != 1.0 else 2/alpha)

    pdf = (b2(alpha/2) * c) * mp.power( x/sigma, (k + 0.5*pfrac - 2.0)) * mp.exp(-a(alpha/2) * mp.power(x/sigma, pfrac))
    return float(pdf)

# -----------------------------------------------------------------------

@lru_cache(maxsize=100)
def frac_chi2_mean(alpha, k, theta=0.0, loc=0.0, scale=1.0):
    # X ~ f(x), then Y = X^2 ~ f(sqrt(y)) / (2*sqrt(y))
    alpha = float(alpha)
    k = float(k)
    k_sign = np.sign(k)  # type: ignore
    assert k != 0, Exception(f"ERROR: k cannot be zero in frac_chi_mean")

    g = (alpha - theta) / (2.0 * alpha)  # instead of 0.5
    sigma = fcm_sigma(alpha, k, theta)
    assert sigma > 0

    if k > 0:  return frac_gamma(alpha=alpha*g, sigma=sigma**2,   d=(k-1)/2, p = alpha/2,  loc=loc, scale=scale)
    if k < 0:  return frac_gamma(alpha=alpha*g, sigma=1/sigma**2, d=k/2,     p = -alpha/2, loc=loc, scale=scale)
    raise Exception(f"ERROR: k is not handled properly")


def fcm2_mellin_transform(s, alpha: float, k: float):
    eps = 1 / alpha
    g = 0.5
    assert k > 0
    sigma = fcm_sigma(alpha, k)
    gamma_k = eps / g if k == 1.0 else gamma(g * (k-1.0)) / gamma(eps * (k-1.0))
    sigma_const = sigma**(2*s-2.0)
    gamma_ks = g / eps if 2*s == 3.0-k else gamma(eps*2 * (s+k/2-3.0/2)) / gamma(s+k/2-3.0/2) 
    return sigma_const * gamma_k * gamma_ks 


def fcm2_moment(n, alpha, k, theta=0.0):
    return fcm_moment(2.0*n, alpha, k, theta)


class FCM2:
    # this is primarily to facilitate the testings, especially for the Mellin transform
    def __init__(self, alpha, k):
        self.alpha = alpha
        self.k = k

        self.c = 2.0 - k + 0.5 if k > 0 else abs(k) + 1.0 - 0.5
        self.fcm2 = frac_chi2_mean(alpha, k)

    def pdf(self, x): 
        return self.fcm2.pdf(x)  # type: ignore
   
    def mellin_transform(self, s): return fcm2_mellin_transform(s, self.alpha, self.k)  # simple wrapper

    def pdf_by_mellin(self, x):
        assert x > 0
        return pdf_by_mellin(x, lambda s: self.mellin_transform(s), c=self.c)

    def pdf_by_wright_f(self, x):
        assert x > 0
        C = fcm_normalization_constant(self.alpha, self.k)
        sigma = fcm_sigma(self.alpha, self.k)
        z = x / sigma**2
        return C/ (2.0 * sigma**2) * z**(self.k/2 - 1.5) * wright_f_fn_by_levy_asymp(z**(self.alpha/2), alpha=self.alpha/2)

    def cdf_by_mellin(self, x):
        assert x > 0
        def _cdf_mellin(s):
            return -1/s * fcm2_mellin_transform(s+1, alpha=self.alpha, k=self.k)
        return pdf_by_mellin(x, _cdf_mellin, c=self.c - 1.0)
    
    def pdf_by_fcm(self, x):
        x2 = np.sqrt(x)
        return frac_chi_mean(self.alpha, self.k).pdf(x2) / (2.0 * x2)  # type: ignore

    def cdf_by_gamma_star(self, x):
        assert self.k > 0
        z = x / fcm_sigma(self.alpha, self.k)**2
        k2 = (self.k - 1.0 + self.alpha) / 2.0
        a2 = self.alpha / 2.0
        return z**k2 * frac_gamma_star(s=k2/a2, x=z**a2, alpha=a2)

    def moment_by_mellin(self, n):
        alpha = self.alpha 
        k = self.k
        assert k > 0
        sigma = fcm_sigma(alpha, k)
        gamma_k = 2/alpha if k == 1.0 else gamma((k-1.0)/2) / gamma((k-1.0)/alpha)
        sigma_const = sigma**(2*n)
        gamma_ks = alpha/2 if 2*n == 1.0-k else gamma(2/alpha * (n+k/2-1.0/2)) / gamma(n+k/2-1.0/2) 
        return sigma_const * gamma_k * gamma_ks
        
    @lru_cache(maxsize=10)
    def moment(self, n) -> float:
        return fcm2_moment(n, self.alpha, self.k)
    
    @lru_cache(maxsize=2)
    def calc_stats_from_moments(self):
        moments = pd.Series({
            n: self.moment(n) for n in range(5)
        })
        return calc_stats_from_moments(moments)

    @lru_cache(maxsize=2)
    def mean(self):
        return self.calc_stats_from_moments()['mean']

    @lru_cache(maxsize=2)
    def variance(self):
        return self.calc_stats_from_moments()['var']
    
    @lru_cache(maxsize=2)
    def skewness(self):
        return self.calc_stats_from_moments()['skew']
    
    @lru_cache(maxsize=2)
    def kurtosis(self):
        # this is excess kurtosis (normal kurtosis is 0.0)
        return self.calc_stats_from_moments()['kurtosis']

    def pdf_multiplied_by(self, x, m):
        # this is analytic formula when pdf(x) is multipliied by x^m
        assert self.k > 0
        sigma_k = fcm_sigma(self.alpha, self.k)
        sigma_k_m = fcm_sigma(self.alpha, self.k + 2*m)
        Q = (sigma_k_m / sigma_k)**2

        C_k = fcm_normalization_constant(self.alpha, self.k)
        C_k_m = fcm_normalization_constant(self.alpha, self.k + 2*m)
        const = sigma_k**(2*m) * (C_k / C_k_m) * Q        

        y = x * Q
        return const * frac_chi2_mean(self.alpha, self.k + 2*m).pdf(y)  # type: ignore


# -----------------------------------------------------------------------
def fcm2_hat(alpha, k):
    # X ~ f(x), then Y = X^2 ~ f(sqrt(y)) / (2*sqrt(y))
    alpha = float(alpha)
    k = float(k)
    assert k > 0
    sigma = 0.5
    return frac_gamma(alpha=alpha/2, sigma=sigma**2, d=(k-1)/2, p = alpha/2)


def fcm2_hat_mellin_transform(s, alpha: float, k: float):
    eps = 1 / alpha
    g = 0.5
    assert k > 0
    sigma_const = 2**(2.0 - 2*s)
    _gamma1 = lambda c: gamma(c * (k-1.0))
    _gamma2 = lambda c: gamma(c * (s+k/2-3.0/2))

    gamma_k  = eps / g if k == 1.0 else _gamma1(g) / _gamma1(eps)
    gamma_ks = g / eps if 2*s == 3.0-k else _gamma2(eps*2) / _gamma2(1.0) 
    return sigma_const * gamma_k * gamma_ks 


# -----------------------------------------------------------------------
# Z = X^2/a + Y^2/b, X and Y are standard normal with cor{X,Y} = rho
# individually, X^2 and Y^2 are chi2(1)
# this is needed to the quadratic form of the 2D adaptive distribution

def chi11_eigen(a, b, rho):
    # inverse of the eigen values from [[1, rho], [rho, 1]] * [[1/a, 0], [0, 1/b]] 
    denom = 2.0 * (1.0 - rho**2)
    assert 0 < denom <= 2
    a1 = (a + b + np.sqrt((a - b)**2 + 4 * rho**2 * a * b)) / denom
    b1 = (a + b - np.sqrt((a - b)**2 + 4 * rho**2 * a * b)) / denom
    return a1, b1


class chi2_11_gen(rv_continuous):

    def _pdf(self, x, a, b, rho, *args, **kwargs):
        # handle array form
        if not isinstance(a, float):
            assert len(a) == len(x), f"ERROR: len of alpha and x"
            if len(x) == 1:  # trvial case
                return self._pdf(x[0], a=a[0], b=b[0], rho=rho[0])
            
            df = pd.DataFrame(data={'x': x, 'a': a, 'b': b, 'rho': rho})
            df['pdf'] = df.parallel_apply(lambda row: self._pdf(row['x'], a=row['a'], b=row['b'], rho=row['rho']), axis=1)  # type: ignore
            return df['pdf'].tolist()

        # integral form
        assert isinstance(x, float)
        assert isinstance(a, float)
        assert isinstance(b, float)
        assert isinstance(rho, float)
        assert a > 0
        assert b > 0
        assert abs(rho) <= 1.0
        if x < 0: return 0.0

        if rho != 0:
            a, b = chi11_eigen(a, b, rho)   
        z1 = (b+a)*x/4
        z2 = abs(b-a)*x/4
        if z2 < 60.0:
            return np.sqrt(a*b)/2 * np.exp(-z1) * iv(0, z2)
        else:
            return np.sqrt(a*b)/2 * np.exp(z2-z1) / np.sqrt(2*np.pi*z2)

    def _cdf(self, x, a, b, rho, *args, **kwargs):
        # handle array form
        if not isinstance(a, float):
            if len(a) == 1 and len(x) > 1:
                n = len(x)
                return self._cdf(x, a=np.repeat(a,n), b=np.repeat(b,n), rho=np.repeat(rho,n))
            assert len(a) == len(x), f"ERROR: len of a and x mismatch: {len(a)} != {len(x)}"
            if len(x) == 1:  # trvial case
                return self._cdf(x[0], a=a[0], b=b[0], rho=rho[0])
            
            df = pd.DataFrame(data={'x': x, 'a': a, 'b': b, 'rho': rho})
            df['cdf'] = df.parallel_apply(lambda row: self._cdf(row['x'], a=row['a'], b=row['b'], rho=row['rho']), axis=1)  # type: ignore
            return df['cdf'].tolist()

        # integral form
        assert isinstance(x, float)
        assert isinstance(a, float)
        assert isinstance(b, float)
        assert isinstance(rho, float)
        assert a > 0
        assert b > 0
        assert abs(rho) <= 1.0
        if x <= 0: return 0.0

        def _kernel(x: float):
            return self._pdf(x, a=a, b=b, rho=rho)  # type: ignore

        return quad(_kernel, 0, x, limit=10000)[0]

    def _argcheck(self, *args, **kwargs):
        # Customize the argument checking here
        a = args[0]
        b = args[1]
        rho = args[2]

        return (
            a > 0  # scale
            and b > 0  # scale
            and abs(rho) <= 1.0  # correlation
        )

    def _rvs(self, a, b, rho, *args, **kwargs):
        size = kwargs.get('size', 1)
        a = float(a)
        b = float(b)
        rho = float(rho)
        corr = np.array([[1.0, rho], [rho, 1.0]])
        return np.array([ (x0**2/a + x1**2/b) for x0, x1 in multivariate_normal(cov=corr).rvs(size=size)]) # type: ignore


# TODO it needs moment formula

chi2_11 = chi2_11_gen(name="ch2(1,1)", shapes="a, b, rho")


def chi2_11_moment(n, a, b, rho):
    c2 = chi2_11(a, b, rho)
    def _kernel(x):
        return x**n * c2.pdf(x)  # type: ignore
    return quad(_kernel, 0, np.inf, limit=10000)[0]

