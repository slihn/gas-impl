import numpy as np 
import pandas as pd
import mpmath as mp
from functools import lru_cache
from scipy.stats import rv_continuous, levy_stable, gengamma
from scipy.special import gamma, loggamma
from scipy.special import sici 
from scipy.integrate import quad, quadrature
from pandarallel import pandarallel  # type: ignore

pandarallel.initialize(verbose=1)

from .wright import wright_fn, levy_stable_extremal
from .frac_gamma_dist import frac_gamma, fg_log_normalization_constant, fg_mellin_transform, fg_normalization_constant, fg_moment, fg_pdf_large_x
from .frac_gamma_dist import  fg_q_by_f, fg_mu_by_f, fg_mu_by_m, fg_mu_by_m_series, fg_mu_at_half_alpha


# Note: GSC is the generalized stable count distribution
# It is renamed to the fractional gamma distribution in 10/2025

# pyright: reportGeneralTypeIssues=false

# --------------------------------------------------------------------------------
def stable_count_pdf_small_x(x, alpha):
    # this is only good for large alpha
    assert alpha > 0 and alpha < 1.0
    q = alpha / gamma(1.0/alpha+1) / gamma(1.0-alpha)
    return q * x**alpha


def stable_count_pdf_wright(x, alpha, max_n=10):
    # this is a replacement of small x asymptotic, especially for small alpha
    assert alpha > 0 and alpha < 1.0
    q = 1.0 / gamma(1.0/alpha+1)
    return q * wright_fn(-x**alpha, lam=-alpha, mu=0.0, max_n=max_n)


def stable_count_pdf_large_x(x, alpha):
    # this formula doesn't work for small alpha
    afrac = alpha/(1.0-alpha)
    a = (1.0-alpha) * alpha**afrac
    b = alpha**(0.5/(1.0-alpha)) / gamma(1.0/alpha+1) / np.sqrt((1.0-alpha) * 2.0*np.pi)
    return b * x**(0.5*afrac) * np.exp(-a * x**afrac)


def stable_count_moment(n, alpha):
    return gamma((n+1)/alpha) / gamma(n+1) / gamma(1/alpha)


# https://github.com/scipy/scipy/blob/v1.8.0/scipy/stats/_continuous_distns.py
# above URL contains many examples how real distributions are implemented


class stable_count_gen(rv_continuous):

    @staticmethod 
    def q(alpha):
        q1 = np.cos(alpha * np.pi/2)
        q2 = np.sin(-alpha * np.pi/2)
        return (q1, q2)
    
    @staticmethod
    @lru_cache(maxsize=100)
    def rv_stable_one_sided(alpha):
        return levy_stable_extremal(alpha)

    def _munp(self, n, alpha, *args, **kwargs):
        # https://github.com/scipy/scipy/issues/13582
        return stable_count_moment(n, alpha)

    def _pdf(self, x, alpha, *args, **kwargs):
        if isinstance(alpha, float):
            assert x >= 0
            if x == 0.0: return 0.0  # N(0) = 0
            rvl = self.rv_stable_one_sided(alpha)
            return rvl.pdf(1.0/x) / x / gamma(1/alpha+1)
        else:
            assert len(alpha) == len(x), f"ERROR: len of alpha and x"
            return [self._pdf(x1, alpha=a1) for x1, a1 in zip(x, alpha)]

    def ccdf_int(self, x, alpha, method="quad"):
        # Note: maybe we don't need this here!
        # broadcast x and alpha
        q1, q2 = self.q(alpha)
        c = 2.0 / np.pi / gamma(1.0/alpha + 1)
        fn = lambda t: np.exp(-q1 * np.power(t, alpha)) * np.sin(-q2 * np.power(t, alpha)) * sici(t/x)[0] * c
        if method == "quad":
            rs = quad(fn, a=0, b=np.inf, limit=1000)
            return rs[0]
        if method == "gaussian":
            rs = quadrature(fn, a=0, b=1000+x, maxiter=1000)
            return rs[0]
        raise Exception("ERROR: Unknown integration method")


stable_count = stable_count_gen(name="stable count", a=0, shapes="alpha")


def wright_f_fn_by_sc(x, alpha: float):
    nu = x**(1/alpha)
    return stable_count(alpha).pdf(nu) * gamma(1.0/alpha + 1)


###################################################################################
#
# generalized stable count distribution: the following functions are just wrappers for backward compatibility
#
def gsc_normalization_constant(alpha, sigma, d, p):  return fg_normalization_constant(alpha, sigma, d, p)

def gsc_log_normalization_constant(alpha, log_sigma, d, p):  return fg_log_normalization_constant(alpha, log_sigma, d, p)

def gsc_moment(n, alpha, sigma, d, p):  return fg_moment(n, alpha, sigma, d, p)

def gsc_mellin_transform(s, alpha: float, sigma: float, d: float, p: float):  return fg_mellin_transform(s, alpha, sigma, d, p)

# ----------------------------------------------------
# mu for RV, various implementations
def gsc_q_by_f(z, dz_ratio, alpha):  return fg_q_by_f(z, dz_ratio, alpha)

def gsc_mu_by_f(x, dz_ratio, alpha, sigma, d, p):  return fg_mu_by_f(x, dz_ratio, alpha, sigma, d, p)

def gsc_mu_by_m(x, dz_ratio, alpha, sigma, d, p):  return fg_mu_by_m(x, dz_ratio, alpha, sigma, d, p)

def gsc_mu_by_m_series(x, alpha, sigma, d, p):  return fg_mu_by_m_series(x, alpha, sigma, d, p)

def gsc_mu_at_half_alpha(x, sigma, d, p):  return fg_mu_at_half_alpha(x, sigma, d, p)

# ----------------------------------------------------
def gsc_pdf_large_x(x, alpha, sigma, d, p):  return fg_pdf_large_x(x, alpha, sigma, d, p)

# ----------------------------------------------------
# this RV is just a legacy wrapper, don't use it for new codes
gen_stable_count = frac_gamma
###################################################################################


def mainardi_wright_fn_in_gsc(alpha: float, scale: float = 1.0):
    # this is treating the mainardi function as the pdf of a distribution
    # among other things, this can be used to get the CDF of the distribution
    # but this is quite slow for large scale usage !!!
    return gen_stable_count(alpha=alpha, sigma=1.0, d=0.0, p=1.0, scale=scale)


# -----------------------------------
# constructors of legacy distributions
# -----------------------------------


class stable_vol_gen(rv_continuous):

    def _munp(self, n, alpha):
        # https://github.com/scipy/scipy/issues/13582
        # mu = self._munp(1, *goodargs)
        if n != -1.0:
            return gamma((n+1)/alpha) / gamma((n+1)/2) / gamma(1/alpha) * np.sqrt(np.pi) * np.power(2.0, -n/2)
        else:
            return 1.0 / 2 / gamma(1/alpha+1) * np.sqrt(np.pi) * np.power(2.0, -n/2)

    def _pdf(self, x, alpha):
        if isinstance(alpha, float):
            assert 0 < alpha <= 2.0
            rv_sc = stable_count(alpha/2.0)
            c = np.sqrt(np.pi*2) * gamma(2/alpha+1) / gamma(1/alpha+1)
            return rv_sc.pdf(2.0 * x*x) * c
        else:
            assert len(alpha) == len(x), f"ERROR: len of alpha and x"
            return [self._pdf(x1, alpha=a1) for x1, a1 in zip(x, alpha)]


stable_vol = stable_vol_gen(name="stable vol", a=0, shapes="alpha")


def sv_mu_by_f(x, dz_ratio, alpha):
    return gsc_mu_by_f(x, dz_ratio, alpha=alpha/2, sigma=1.0/np.sqrt(2), d=1.0, p=alpha)
