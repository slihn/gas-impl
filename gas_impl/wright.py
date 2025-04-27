from functools import lru_cache
import numpy as np 
import mpmath as mp
import tsquad

from scipy.special import gamma
from scipy.integrate import quad
from scipy.stats import levy_stable
from typing import Union, List, Optional


# --------------------------------------------------------------------------------
# this handles negative z using reflection rule
def mp_gamma(z):
    if z > 0: return mp.gamma(z) 
    if z < 0: return mp.pi / mp.gamma(1-z) / mp.sin(z * mp.pi)
    return mp.inf


# --------------------------------------------------------------------------------
def wright1(n: int, x: float, lam: float, mu: float) -> float:
        # the n-th term of Wright Fn
        return np.power(x, n) / gamma(n+1.0) / gamma(lam*n + mu)  # type: ignore


def wright_fn(x: Union[float, int, List], lam: float, mu: float, max_n: int=40, start: int=0):
    if isinstance(x, int):
        x = 1.0 * x
    if isinstance(x, float):
        p = np.array([wright1(k, x, lam, mu) for k in range(start, max_n+1)])
        return sum(p)

    if len(x) >= 1:
        return [wright_fn(x1, lam, mu, max_n=max_n, start=start) for x1 in x]
    raise Exception(f"ERROR: unknown x: {x}")


def wright_m_fn(x, alpha):
    # bad convergence
    return 1.0/(alpha*x) * wright_fn(-x, -alpha, 0)


def mainardi_wright_fn(x, alpha, max_n: int=80):
    # good convergence
    # but this might need mp version to do a really good job, especially when alpha and/or x is large
    return wright_fn(-x, -alpha, 1.0-alpha, max_n=max_n)


def mainardi_wright_fn_cdf(x, alpha, max_n: int=80):
    # \int_0^x m_wright_fn(t, alpha) dt = 1 - wright_fn(-x, -alpha, 1.0)
    # good convergence, typically well behaved when alpha < 0.5 and x < 1.0
    # but this might need mp version to do a really good job, especially when alpha and/or x is large
    # for alpha > 0.5, use the cdf of the extremal stable distribution below
    return 1.0 - wright_fn(-x, -alpha, 1.0, max_n=max_n)  # type: ignore


def mainardi_wright_fn_cdf_by_levy(x, alpha):
    assert 0.5 <= alpha < 1.0
    alpha2 = 1.0 / alpha
    mw_rv = levy_stable_extremal(alpha2)
    return (mw_rv.cdf(x) - mw_rv.cdf(0)) * alpha2


# --------------------------------------------------------------------------------
# this uses tsquad to carry out Prodanov integral of the M-Wright function
# (11) and Theorem 1 of Prodanov (2023): Computation of the Wright function from its integral representation
def wright_mainardi_fn_ts(x, alpha) -> float:
    if alpha == 0: return np.exp(-x)
    assert alpha > 0 and alpha < 1.0
    assert isinstance(x, float)
    b = ()
    return wright_fn_ts(x, alpha, b = 1.0 - alpha)


def wright_mainardi_fn_cdf_ts(x, alpha) -> float:
    # doing poorly when alpha is too large or too small (< 0.1), and/or x is too large (x > 100)
    # max(x) is getting smaller when alpha > 0.5, you can use mainardi_wright_fn_cdf_by_levy() instead
    assert alpha > 0 and alpha < 1.0
    assert isinstance(x, float)
    return -1.0 * wright_fn_ts(x, alpha, b=1.0)


def wright_fn_ts(x, alpha, b) -> float:
    # W_{-alpha, b}(-x) 
    assert 0 <= b <= 1.0, f"ERROR: b={b} is not in [0,1], whhich is required by row 1 of the Table in Theorem 1"
    a = -alpha
    z = -x
    pi = np.pi
    lower = 0.0  # fn(r) is singular at r=0, but it does not seem to matter for the integral
    
    def fn(r):
        y1 = 1.0/(pi * r**b)
        zra = z / r**(a)
        y2 = np.sin( np.sin(a*pi) * zra + pi * b )
        y3 = np.exp( np.cos(a*pi) * zra - r )
        return y1 * y2 * y3

    rs = tsquad.QuadTS(f=fn).quad(lower, np.inf)
    return rs.I 


# --------------------------------------------------------------------------------
class M_Wright_One_Sided:
    # one-sided M-Wright function, as a distribution, mainly for the purpose of CDF
    def __init__(self, alpha: float):
        assert 0 < alpha < 1.0
        self.alpha = alpha
    
    def pdf(self, x):
        # a lot of nuances have been handled inside the by_levy function
        # this is just a dummy wrapper
        return wright_m_fn_by_levy(x, self.alpha)
    
    def cdf(self, x):
        assert x >= 0  # one-sided
        if self.alpha < 0.5 and x < 1.0:
            return mainardi_wright_fn_cdf(x, self.alpha)  # this is very fast
        if self.alpha >= 0.1 and self.alpha < 0.5:
            x = 100.0 if x > 100.0 else x 
            return wright_mainardi_fn_cdf_ts(x, self.alpha)
        if self.alpha >= 0.5:
            # levy_stable can handle everything above 0.5 every effectively
            return mainardi_wright_fn_cdf_by_levy(x, self.alpha)
        raise Exception(f"ERROR: CDF unsupported for alpha={self.alpha} and x={x}")

    def moment(self, n):
        return gamma(n + 1.0) / gamma(self.alpha * n + 1.0)


# --------------------------------------------
def wright_m_fn_rescaled_by_levy(x, g: float):
    if g == 0.0: return x * 0.0  # to preserve the shape of x
    assert g > 0.0  # g can not be zero
    return g**(1-g) * wright_m_fn_by_levy(x / (g**g), alpha=g)  # type: ignore

def wright_m_fn_rescaled_f0(g):
    # should be wright_m_fn_rescaled_by_levy(0, g)
    return np.sin(g*np.pi) * gamma(g) * g**(1-g) / np.pi

def wright_m_fn_rescaled_f1(g):
    # should be slope of wright_m_fn_rescaled_by_levy(0, g)
    return -np.sin(2*g*np.pi) * gamma(2*g) * g**(1-2*g) / np.pi

def wright_m_fn_rescaled_psi(g, c=0):
    # this is a useful wrapper function to compute f0() and f1()
    return np.sin((g+c) * np.pi) * gamma(g+c) / np.pi


# --------------------------------------------
class M_Wright_Rescaled_Dist:
    # rescaled M-Wright function, as a two-sided distribution
    def __init__(self, g: float):
        assert 0 < g < 1.0
        self.g = g
    
    def unadjusted_pdf(self, x):
        if x <= 0: return wright_m_fn_rescaled_by_levy(-x, 1.0-self.g)  # reflection
        return wright_m_fn_rescaled_by_levy(x, self.g)
    
    def pdf(self, x):
        g = self.g
        assert isinstance(x, float)
        if x >= 0: 
            return wright_m_fn_rescaled_by_levy(x, g) / self.A_plus()
        else:
            return wright_m_fn_rescaled_by_levy(-x/self.Sigma(), 1-g) * self.Psi() / self.Sigma() / self.A_plus()

    @lru_cache(maxsize=4)
    def _h(self, c=0.0):
        # c is really for 0 and 1/2
        g = self.g
        return gamma(g+c) * gamma(g-c) / np.pi * np.sin((g+c)*np.pi)

    @lru_cache(maxsize=2)
    def _h_star(self): 
        g = self.g
        return 4**(2*g-1) * self._h(c=0.5)

    @lru_cache(maxsize=2)
    def Sigma(self):
        g = self.g
        c = (g*(1-g))**g 
        return -c/(1-g) / self._h_star()

    @lru_cache(maxsize=2)
    def Psi(self):
        g = self.g
        return -g/(1-g) * self._h() / self._h_star()

    @lru_cache(maxsize=2)
    def A_plus(self):
        g = self.g
        return g + self.Psi() * (1-g)

    def moment(self, n):
        return np.nan  # TODO
    

# --------------------------------------------------------------------------------

def mainardi_wright_fn_slope(x, alpha, max_n: int=80):
    # good convergence
    # but this might need mp version to do a really good job
    return -1*wright_fn(-x, -alpha, 1.0-2*alpha, max_n=max_n)


def wright_f_fn(x, alpha):
    # bad convergence
    return wright_fn(-x, -alpha, 0)


# --------------------------------------------------------------------------------
def wright_fn_mellin_transform(s, lam: float, mu: float):
    return gamma(s) / gamma(mu - lam * s)


def wright_f_fn_mellin_transform(s, alpha: float):
    return gamma(s) / gamma(alpha * s)


def wright_m_fn_mellin_transform(s, alpha: float):
    return gamma(s) / gamma((1.0-alpha) + alpha * s)


def wright_m_fn_rescaled_mellin_transform(s, g: float):
    # g**(1-g) * M_g(x / (g**g))
    return gamma(s) / gamma((1.0-g) + g * s) * g**(g*s + 1.0-g)


def norm_mellin_transform(s):
    # normal distribution
    return gamma(s/2) * 2**((s-3)/2) / np.sqrt(np.pi)


def wright_m_fn_moment(n, alpha: float):
    # s = n+1, this is: int_0^inf x^n wright_m_fn(x, alpha) dx
    return gamma(n+1.0) / gamma((1.0-alpha) + alpha * (n+1.0))


# --------------------------------------------------------------------------------
def levy_stable_extremal(alpha: float, scale: Optional[float] = None, scale_multiplier: float = 1.0):
    assert 0 < alpha <= 2.0
    if scale is None:
        theta = alpha if alpha <= 1.0 else (2.0 - alpha)
        scale = np.power(np.cos(theta * np.pi / 2.0), 1.0/alpha)  # type: ignore
    assert isinstance(scale, float), f"ERROR: scale={scale} is not float for alpha={alpha}"
    scale = scale * scale_multiplier
    assert scale > 0
    beta = 1.0 if alpha <= 1.0 else -1.0
    return levy_stable(alpha, beta=beta, loc=0, scale=scale)


def wright_f_fn_by_levy(x, alpha: float):
    # this is faster than wright_f_fn_by_sc, if you need speed
    assert 0 < alpha < 1.0
    rv = levy_stable_extremal(alpha)  # See (F.48) of Mainardi's Appendix F
    y = x**(-1.0/alpha)
    return rv.pdf(y) * y  # type: ignore


def log_wright_f_fn_by_levy(log_x, alpha: float):
    # this is faster than wright_f_fn_by_sc, if you need speed
    assert 0 < alpha < 1.0
    rv = levy_stable_extremal(alpha)  # See (F.48) of Mainardi's Appendix F
    y = np.exp(-log_x / alpha)
    log_y = (-1.0/alpha) * log_x
    return rv.logpdf(y) + log_y  # type: ignore


def wright_m_fn_by_levy(x, alpha: float):
    assert 0 < alpha < 1.0, f"ERROR: alpha={alpha} is not in (0,1)"
    if x == 0: return 1.0 / gamma(1.0 - alpha)
    if abs(x) <= 0.01: return mainardi_wright_fn(x, alpha, max_n=7)  # levy_stable has issue with small x
    if alpha >= 0.5:
        alpha2 = 1.0 / alpha
        rv = levy_stable_extremal(alpha2)
        # See (F.49) of Mainardi's Appendix F
        return rv.pdf(x) * alpha2  # type: ignore

    assert x >= 0, f"ERROR: x={x} is negative. Not allowed for alpha < 0.5 (alpha={alpha})"
    return wright_f_fn_by_levy(x, alpha) / x / alpha


class Skew_MWright_Dist:
    def __init__(self, g: float):
        self.g: float = g
        self.g_sqrt: float = np.sqrt(self.g)
        assert 0.5 <= self.g < 1.0
        self.rv = levy_stable_extremal(alpha = 1.0/g, scale_multiplier = self.g_sqrt)
    
    def pdf(self, x, correct_small_x = True):
        if correct_small_x == True and abs(x/self.g_sqrt) <= 0.01:
            return mainardi_wright_fn(x/self.g_sqrt, alpha=self.g, max_n=7) * self.g_sqrt  # type: ignore
        return self.rv.pdf(x)  # type: ignore
    
    def pdf_by_fn(self, x):
        return wright_m_fn_by_levy(x/self.g_sqrt, alpha=self.g) * self.g_sqrt  # type: ignore

    def cdf(self, x):
        return self.rv.cdf(x)


# --------------------------------------------------------------------------------
def wright_q_fn(x, alpha):
    f = wright_f_fn(x, alpha)
    return wright_fn(-x, -alpha, -1.0) / (-1.0 * f)  # type: ignore

 
def mittag_leffler_fn(x: Union[float, int, List], alpha: float, beta: float=1.0, max_n: int=40, start: int=0):
    
    def _mlf_item(k):
        return x**k / gamma(alpha*k + beta)
    
    if isinstance(x, int):
        x = 1.0 * x
    if isinstance(x, float) or isinstance(x, complex):
        p = np.array([_mlf_item(k) for k in range(start, max_n+1)])
        return sum(p)

    if len(x) >= 1:
        return [mittag_leffler_fn(x1, alpha, beta, max_n=max_n, start=start) for x1 in x]
    raise Exception(f"ERROR: unknown x: {x}")
