
import numpy as np
import pandas as pd
from scipy.special import gamma
from scipy.integrate import quad
from .wright import mainardi_wright_fn, wright_f_fn_by_levy
from .hyp_geo import frac_hyp1f1_m, frac_hyp1f1_m_int


# ------------------------------------------------------------------
# gamma star
def frac_gamma_star_by_hyp1f1(s, x, alpha, use_int=False, by_levy=False):
    # the series form of the new gamma star
    g = gamma(alpha*s - alpha + 1.0) / gamma(s+1)
    if use_int:
        p = frac_hyp1f1_m_int(-x, alpha, s, s+1, by_levy=by_levy)
    else:
        p = frac_hyp1f1_m(-x, alpha, s, s+1)
    return p * g


def frac_gamma_star_by_m(s, x, alpha):
    # this is primarily used for alpha near zero, or small x, using the series form
    if x == 0: return 0.0
    def fn1(t): 
        return t**(s-1) * mainardi_wright_fn(x * t, alpha)
    p = quad(fn1, 0, 1, limit=200)[0]
    g = gamma(alpha*s - alpha + 1.0) / gamma(s)
    return p * g


# but this is slower
def frac_gamma_star_by_f(s, x, alpha):
    if x == 0: return 0.0
    def fn1(t): 
        return t**(s-2) * wright_f_fn_by_levy(x * t, alpha) / x
    p = quad(fn1, 0, 1, limit=1000)[0]
    g = gamma(alpha * (s - 1.0)) / gamma(s - 1.0)
    return p * g


def frac_gamma_star(s, x, alpha):
    # this is wrapper on top of frac_hyp1f1_int, since it can support widest ranges
    if x == 0: return 0.0
    if alpha < 0.05: return frac_gamma_star_by_hyp1f1(s, x, alpha, use_int=True)
    return frac_gamma_star_by_hyp1f1(s, x, alpha, use_int=True, by_levy=True)  # but this is slower


# ------------------------------------------------------------------------
# I changed my mind, don't hink frac_gamma_inc in this form is a good idea
# it is too complicated than necessary
def frac_gamma_inc_by_m(s, x, alpha):
    if x == 0: return 0.0
    if x == 0: return 0.0
    def fn1(t): 
        return t**(s-1) * mainardi_wright_fn(2 * t**0.5, alpha)
    p = quad(fn1, 0, x, limit=200)[0]
    g = np.sqrt(np.pi) * gamma(2*alpha*s + 1.0 - alpha) / gamma(s) / gamma(s + 0.5)
    return p * g


def frac_gamma_inc(s, x, alpha):
    if x == 0: return 0.0
    x2 = 2.0 * x**0.5
    s2 = 2.0 * s
    return frac_gamma_star(s2, x2, alpha) * x2**s2

