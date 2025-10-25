from functools import lru_cache
from math import isfinite
import numpy as np
import pandas as pd
import mpmath as mp
from scipy.special import gamma
from scipy.stats import gengamma
from scipy.optimize import root_scalar

from .utils import make_list_type


# ----------------------------------------------------------------
def wright_m_fn_moment(n, alpha: float) -> float:
    return gamma(n+1.0) / gamma(alpha * n + 1.0)  


def wright_m_fn_mean(alpha: float) -> float:
    return wright_m_fn_moment(1.0, alpha)


def wright_m_fn_std(alpha: float) -> float:
    var = wright_m_fn_moment(2.0, alpha) - wright_m_fn_moment(1.0, alpha)**2
    return np.sqrt(var)


# ----------------------------------------------------------------
# ON THE ASYMPTOTICS OF WRIGHT FUNCTIONS OF THE SECOND KIND (2023)
# Richard B. Paris, Armando Consiglio, Francesco Mainardi


def gengamma_from_gg(a, d, p):
    return gengamma(a=d/p, c=p, scale=a)


def wright_m_fn_asymp_gg(x, alpha):
    if isinstance(x, (list, np.ndarray, pd.Series)):
        rs = np.array([wright_m_fn_asymp_gg(xi, alpha) for xi in x])  # type: ignore
        return make_list_type(rs, x)

    # Section 3.3 of the book
    assert isinstance(x, float)
    alpha = float(alpha)
    if alpha == 0: return np.exp(-x)  # this is precise
    assert alpha < 1.0 and alpha > 0.0
    y = x * alpha
    # singular at alpha = 1, more precise at x > 1 and m is small, e.g. m < 1e-4
    p = 1.0/(1-alpha)
    d = p/2  # (alpha-0.5) / (1-alpha) + 1
    sigma = (alpha * p)**(1/p)
    # alternatively, gg = (p/gamma(0.5)/ sigma) * (y/sigma)**(d-1) * np.exp(-(y/sigma)**p)
    gg = gengamma_from_gg(a=sigma, d=d, p=p).pdf(y)  # type: ignore
    z = (alpha / 2)**(0.5) * gg
    return z


def wright_m_fn_find_x_by_asymp_gg(target, alpha):
    # x must be on the right side of the mean
    assert target > 1e-20
    mn = wright_m_fn_moment(alpha, 1)
    max_bound = 20.0
    if alpha > 0.99:
        min_fn_val = 1e-50  # just a small value that float64 can calculate
        max_bound = wright_m_fn_asymp_gg_find_x_by_step(min_fn_val, alpha)

    def _func1(x):
        return wright_m_fn_asymp_gg(x, alpha) - target 

    result = root_scalar(_func1, bracket=[mn, max_bound], method='brentq')
    if result.converged:
        return result.root
    else:
        return np.nan


def wright_m_fn_asymp_gg_find_x_by_step(target, alpha, step=None):
    # primarily for alpha > 0.99, we need to know the max bound of asymp_gg
    # we want to be able to handle alpha=0.999
    if step is None:
        step = 0.001 if alpha > 0.99 else 0.01
    x = wright_m_fn_mean(alpha)
    while True:
        v = wright_m_fn_asymp_gg(x, alpha)
        if v < target:
            break
        x += step
    return x


# ----------------------------------------------------------------
# below is from Paris 2023, Section 2, sigma is just alpha below
def d2(sigma):
    return 2.0 + 19*sigma + 2*sigma**2

def d3(sigma):
    return (1.0/5) * (556 - 1628*sigma - 9093*sigma**2 - 1628*sigma**3 + 556*sigma**4)

def d4(sigma):
    return (1.0/5) * (4568 + 226668*sigma - 465702*sigma**2 - 2013479*sigma**3
                    - 465702*sigma**4 + 226668*sigma**5 + 4568*sigma**6)

def d5(sigma):
    return (1.0/7) * (2622064 - 12598624*sigma - 167685080*sigma**2 + 302008904*sigma**3
                    + 1115235367*sigma**4 + 302008904*sigma**5 - 167685080*sigma**6
                    - 12598624*sigma**7 + 2622064*sigma**8)

def d6(sigma):
    return (1.0/35) * (167898208 + 22774946512*sigma - 88280004528*sigma**2
                     - 611863976472*sigma**3 + 1041430242126*sigma**4
                     + 3446851131657*sigma**5 + 1041430242126*sigma**6
                     - 611863976472*sigma**7 - 88280004528*sigma**8
                     + 22774946512*sigma**9 + 167898208*sigma**10)


def d(sigma: float, j: int) -> float:
    assert j >= 1
    if j == 1: return 1.0
    if j == 2: return d2(sigma)
    if j == 3: return d3(sigma)
    if j == 4: return d4(sigma)
    if j == 5: return d5(sigma)
    if j == 6: return d6(sigma)

    raise NotImplementedError("Higher order d(s,j) are not implemented.")


@lru_cache(maxsize=100)
def c(sigma: float, j: int) -> float:
    # Paris 2023 (2.4)
    if j == 0: return 1.0
    assert j >= 1
    A = (2.0 - sigma) * (1.0 - 2*sigma)
    B = 2**(3*j) * 3**j * gamma(j+1) * sigma**j
    return A / B * d(sigma,j)


# ----------------------------------------------------------------
def wright_m_fn_asymp_paris(x, alpha, max_j=6):
    if isinstance(x, (list, np.ndarray, pd.Series)):
        rs = np.array([wright_m_fn_asymp_paris(xi, alpha) for xi in x])  # type: ignore
        return make_list_type(rs, x)

    x = float(x)
    alpha = float(alpha)
    # below is from Paris 2023 (2.1)
    kappa = 1 - alpha
    h = alpha**alpha
    # Paris 2023 (2.5)
    assert max_j >= 0 and max_j <= 6
    A = np.sqrt(2*np.pi / alpha) * (alpha/kappa)**alpha

    try:
        X = kappa * (h * x)**(1/kappa)  # x**(1/kappa) is problematic for alpha >= 0.993, 1/kappa = 1/0.007 = 143
    except OverflowError:
        X = np.inf

    if np.isfinite(X):
        C = np.array([c(alpha, j) * (-X)**(-j) for j in range(max_j+1)]).sum()
        return A / (2*np.pi) * X**(alpha-0.5) * np.exp(-X) * C
    else:
        return 0.0

