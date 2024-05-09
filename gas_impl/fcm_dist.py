from math import isnan
import numpy as np 
import pandas as pd
import mpmath as mp
from typing import Union
from functools import lru_cache
from scipy.stats import rv_continuous, levy_stable
from scipy.special import gamma, erf
from scipy.stats import norm
from scipy.integrate import quad, IntegrationWarning
from scipy.optimize import root_scalar


from .wright import wright_fn, mp_gamma
from .stable_count_dist import gen_stable_count, wright_f_fn_by_sc, gsc_q_by_f

# --------------------------------------------------------------------------------
# FCM
def fcm_sigma(alpha: float, k: float):
    return abs(k)**(0.5 - 1/alpha) / np.sqrt(2)


def fcm_moment(n: float, alpha: float, k: float, k_mean=True):
    # k_mean = False is for classic chi distribution
    # we use mpf for cancellation, not for output, but you need to set the precision
    n = mp.mpf(n)
    k = mp.mpf(k)
    alpha = mp.mpf(alpha)
    assert k != 0
    sigma = mp.power(abs(k), 1/mp.mpf(2)-1/alpha) / mp.sqrt(2)
    if k > 0:
        sigma_n = mp.power(sigma, n) if k_mean else mp.power(2.0, -n/2.0)
        c = mp_gamma((k-1)/2) / mp_gamma((k-1)/alpha) if k != 1.0 else 2/alpha
        d = mp_gamma((k+n-1)/alpha) / mp_gamma((k+n-1)/2) if k+n != 1.0 else alpha/2
        return float(sigma_n * c * d)
    if k < 0:
        assert k_mean == True
        sigma_n = mp.power(sigma, -n)
        c = mp_gamma(abs(k)/2) / mp_gamma(abs(k)/alpha)
        d = mp_gamma((abs(k)-n)/alpha) / mp_gamma((abs(k)-n)/2) if n != abs(k) else alpha/2
        return float(sigma_n * c * d)
    raise Exception(f"ERROR: k is not handled properly")


def fcm_q_by_f(z, dz_ratio, alpha):
    z = z**alpha
    if z == 0: z = 0.001
    f = wright_f_fn_by_sc(z, alpha/2)
    assert abs(f) > 0, f"ERROR: z = {z}, f = {f} for alpha {alpha}"
    if dz_ratio is None:
        q = wright_fn(-z, -alpha/2, -1.0) / -f
    else:
        dz = z * dz_ratio
        f_dz =  wright_f_fn_by_sc(z+dz, alpha/2)
        q = (alpha/2 * z) * (f_dz - f)/dz/f + 1
    return q 


def fcm_q_by_gsc_q(z, dz_ratio, alpha):
    # this is just for testing
    return gsc_q_by_f(z**alpha, dz_ratio, alpha/2)


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


@lru_cache(maxsize=100)
def frac_chi_mean(alpha, k):
    alpha = float(alpha)
    k = float(k)
    k_sign = np.sign(k)  # type: ignore
    assert k != 0, Exception(f"ERROR: k cannot be zero in frac_chi_mean")
    sigma = fcm_sigma(alpha, k)
    assert sigma > 0
        
    d = k - (k_sign+1)/2.0  # k-1 if k > 0, else k
    # TODO
    # gen_stable_count(alpha=alpha/2, sigma=k_sigma**(-k_sign), d=d, p = alpha*k_sign)

    if k > 0:  return gen_stable_count(alpha=alpha/2, sigma=sigma, d=k-1, p = alpha)
    if k < 0:  return gen_stable_count(alpha=alpha/2, sigma=1/sigma, d=k, p = -alpha)
    raise Exception(f"ERROR: k is not handled properly")


# alias, make it simple
def fcm(alpha, k):  return frac_chi_mean(alpha, k)


def fcm_inverse(alpha, k):
    sigma = fcm_sigma(alpha, k)
    if k < 0: return gen_stable_count(alpha=alpha/2, sigma=sigma, d=abs(k), p = alpha)
    if k > 0: return gen_stable_count(alpha=alpha/2, sigma=1/sigma, d=-(k-1), p = -alpha)
    raise Exception(f"ERROR: k is not handled properly")

    
def fcm_inverse_pdf(x, alpha, k):
    # this is not meant to be efficient, just used for proof
    c = fcm_moment(1.0, alpha, -k)
    pdf = frac_chi_mean(alpha=alpha, k=-k).pdf(x)  # type: ignore 
    return pdf * x / c  


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
