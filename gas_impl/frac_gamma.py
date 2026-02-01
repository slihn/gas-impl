
from adp_tf.stable_count.wright_levy_asymp import wright_f_fn_by_levy_asymp
import numpy as np
import pandas as pd
from scipy.special import gamma
from scipy.integrate import quad
from .wright import mainardi_wright_fn
from .wright_levy_asymp import wright_m_fn_by_levy_asymp, wright_f_fn_by_levy_asymp
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


def frac_gamma_star_by_m(s, x, alpha, by_levy=False):
    # this is primarily used for alpha near zero, or small x, using the series form
    if x == 0: return 0.0
    _wright_fn = wright_m_fn_by_levy_asymp if by_levy else mainardi_wright_fn
    def fn1(t): 
        return t**(s-1) * _wright_fn(x * t, alpha)
    p = quad(fn1, 0, 1, limit=10000)[0]
    g = gamma(alpha*s - alpha + 1.0) / gamma(s)
    return p * g


def frac_gamma_star_supplementary_by_m(s, x, alpha, a=1.0, b=np.inf):
    # this is primarily used for testing in proof of concept
    if x == 0: return 0.0
    def fn1(t): 
        return t**(s-1) * wright_m_fn_by_levy_asymp(x * t, alpha)
    p = quad(fn1, a=a, b=b, limit=10000)[0]
    g = gamma(alpha*s - alpha + 1.0) / gamma(s)
    return p * g


# but this is slower
def frac_gamma_star_by_f(s, x, alpha):
    if x == 0: return 0.0
    def fn1(t): 
        return t**(s-2) * wright_f_fn_by_levy_asymp(x * t, alpha) / x
    p = quad(fn1, 0, 1, limit=1000)[0]
    g = gamma(alpha * (s - 1.0)) / gamma(s - 1.0)
    return p * g


def frac_gamma_star(s, x, alpha):
    # this is wrapper on top of frac_hyp1f1_int, since it can support widest ranges
    # Note: This has issue when used in frac gamma distr, where p is negative. 
    # when x is small, x^p is very large, x^p hyp1f1(x^p) is unstable
    if x == 0: return 0.0
    if alpha < 0.05: return frac_gamma_star_by_hyp1f1(s, x, alpha, use_int=True)
    return frac_gamma_star_by_hyp1f1(s, x, alpha, use_int=True, by_levy=True)  # but this is slower


def frac_gamma_star_total(s, x, alpha):
    # this is the identity that sums up to 1, otherwise there is a numerical issue in one of them
    # e.g. for alpha = 1, s = 3, when x is larger than 200, frac_gamma_star is losing accuracy
    # on the other hand, large s (s > 5) doesn't work well with small alpha (alpha < 0.1)
    fg = frac_gamma_star(s, x, alpha=alpha)
    fg_supp = frac_gamma_star_supplementary_by_m(s, x, alpha=alpha)
    return (fg + fg_supp) * x**s


# ------------------------------------------------------------------------
# I changed my mind, don't think frac_gamma_inc in this form is a good idea
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

