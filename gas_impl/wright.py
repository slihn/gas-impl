from functools import lru_cache
import numpy as np
import pandas as pd
import mpmath as mp
import tsquad

from scipy.special import gamma
from scipy.optimize import root_scalar
from scipy.stats import levy_stable, norm
from typing import Union, List, Optional

from .utils import make_list_type, calc_elasticity
from .wright_asymp import wright_m_fn_moment, wright_m_fn_find_x_by_asymp_gg


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


def wright_fn(x: Union[float, int, List, np.ndarray, pd.Series], lam: float, mu: float, max_n: int=40, start: int=0):
    if isinstance(x, int):
        x = 1.0 * x
    if isinstance(x, float):
        p = np.array([wright1(k, x, lam, mu) for k in range(start, max_n+1)])
        return sum(p)

    if isinstance(x, (list, np.ndarray, pd.Series)):
        rs = np.array([wright_fn(x1, lam, mu, max_n=max_n, start=start) for x1 in x])  # type: ignore
        return make_list_type(rs, x)
    raise Exception(f"ERROR: unknown x: {x}")



def wright_ratio_fn(x: Union[float, int, List, np.ndarray, pd.Series], lam: float, mu: float, delta: float, max_n: int=40):
    if isinstance(x, (list, np.ndarray, pd.Series)):
        rs = np.array([wright_ratio_fn(x1, lam, mu, delta, max_n=max_n) for x1 in x])  # type: ignore
        return make_list_type(rs, x)

    assert isinstance(x, (float, int)), f"ERROR: x={x} is not float or int"
    f1 = wright_fn(x, lam, mu+delta, max_n=max_n)
    f2 = wright_fn(x, lam, mu, max_n=max_n)
    return f1 / f2  # type: ignore


def wright_fn_elasticity(x: Union[float, int, List, np.ndarray, pd.Series], lam: float, mu: float,
                         max_n: int=40, version: int=1, d_log_x: float=0.00001):
    if isinstance(x, (list, np.ndarray, pd.Series)):
        rs = np.array([wright_fn_elasticity(x1, lam, mu, max_n=max_n, version=version) for x1 in x])  # type: ignore
        return make_list_type(rs, x)

    assert isinstance(x, (float, int)), f"ERROR: x={x} is not float or int"
    if version == 1:
        q = wright_ratio_fn(x, lam, mu, delta=-1.0, max_n=max_n)
        return (q + 1.0 - mu) / lam  # type: ignore
    if version == 2:
        q = wright_ratio_fn(x, lam, mu, delta=lam, max_n=max_n)
        return x * q  # type: ignore
    if version == 3:
        fn = lambda t: wright_fn(t, lam, mu, max_n=max_n)
        return calc_elasticity(fn, x, d_log_x=d_log_x)

    raise Exception(f"ERROR: unknown version={version}")


# --------------------------------------------------------------------------------
def wright_m_fn(x, alpha):
    # bad convergence
    return 1.0/(alpha*x) * wright_fn(-x, -alpha, 0)


def mainardi_wright_fn(x, alpha, max_n: int=80):
    # good convergence
    # but this might need mp version to do a really good job, especially when alpha and/or x is large
    assert 0 <= alpha < 1.0
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
    return wright_fn_ts(x, alpha, b = 1.0 - alpha)


def wright_mainardi_fn_cdf_ts(x, alpha) -> float:
    # this is doing poorly when alpha is too large or too small (< 0.1), and/or x is too large (x > 100)
    # max(x) is getting smaller when alpha > 0.5, you can use mainardi_wright_fn_cdf_by_levy() instead
    # the integral at b = 1 is not stable
    assert alpha > 0 and alpha < 1.0
    assert isinstance(x, float)
    return -1.0 * wright_fn_ts(x, alpha, b=1.0)


# this uses tsquad to carry out Prodanov integral of the Wright function of W_{-alpha, b}(-x) type
# M-Wright is just a special case of it
def wright_fn_ts(x, alpha, b) -> float:
    # calculate W_{-alpha, b}(-x) 
    assert 0 <= b <= 1.0, f"ERROR: b={b} is not in [0,1], whhich is required by row 1 of the Table in Theorem 1"
    a = -alpha
    z = -x
    pi = np.pi
    lower = 0.0  # fn(r) is singular at r=0, but it does not seem to matter for the integral
    
    def fn(r):
        y1 = 1.0/(pi * r**b)
        zra = z / r**(a)  # this causes y2 to oscillate, not good when z is large
        y2 = np.sin( np.sin(a*pi) * zra + pi * b )
        y3 = np.exp( np.cos(a*pi) * zra - r )
        return y1 * y2 * y3

    try:
        rs = tsquad.QuadTS(f=fn, rec_limit=100).quad(lower, np.inf)
        return rs.I
    except Exception as e:
        print(f"ERROR: x={float(x)}, alpha={float(alpha)}, b={float(b)}, {e}")
        return np.nan


# --------------------------------------------------------------------------------
class M_Wright_One_Sided:
    # one-sided M-Wright function, as a distribution, mainly for the purpose of CDF
    def __init__(self, alpha: float):
        assert 0 <= alpha < 1.0
        self.alpha = alpha
    
    def pdf(self, x):
        assert x >= 0  # one-sided
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
    assert 0 < alpha <= 1.0
    rv = levy_stable_extremal(alpha)  # See (F.48) of Mainardi's Appendix F
    y = x**(-1.0/alpha)
    return rv.pdf(y) * y  # type: ignore


def log_wright_f_fn_by_levy(log_x, alpha: float):
    # this is faster than wright_f_fn_by_sc, if you need speed
    assert 0 < alpha <= 1.0
    rv = levy_stable_extremal(alpha)  # See (F.48) of Mainardi's Appendix F
    y = np.exp(-log_x / alpha)
    log_y = (-1.0/alpha) * log_x
    return rv.logpdf(y) + log_y  # type: ignore


def wright_m_fn_by_levy(x, alpha: float):
    assert 0 <= alpha <= 1.0, f"ERROR: alpha={alpha} is not in [0,1]"
    if isinstance(x, (list, np.ndarray, pd.Series)):
        rs = np.array([wright_m_fn_by_levy(x1, alpha) for x1 in x])  # type: ignore
        return make_list_type(rs, x)

    assert isinstance(x, float), f"ERROR: x={x} is not float"
    if x == 0: return 1.0 / gamma(1.0 - alpha)
    if alpha == 1.0:
        # fake the delta function at x=1
        return norm(loc=1.0, scale=0.001).pdf(x) # type: ignore  

    # levy_stable has issue with small x, wright_f_fn_by_levy has issue with small alpha    
    if abs(x) <= 0.01 and alpha != 1.0:
        if alpha <= 0.9: 
            return mainardi_wright_fn(x, alpha, max_n=7)
        else:
            return mainardi_wright_fn(x, alpha)  # type: ignore  # use all terms
    if alpha <= 0.08:
        # 0.08 is where the series and by_levy have about the same precision at large x
        y: float = mainardi_wright_fn(x, alpha)  # type: ignore  # use all terms
        return y if y > 0 else np.nan

    if 0.99 <= alpha <= 0.999 and x <= 0.85:
        # use series expansion for better precision than levy_stable in the high alpha range, but not too close to the delta function
        return mainardi_wright_fn(x, alpha)
    
    assert alpha <= 1.0, f"ERROR: alpha={alpha} is not less than 1.0"
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


# utilities to compute mode of M-Wright function
# mode is zero when alpha <= 0.5, and needs special handling when alpha is close to 0.5 from above
def wright_m_fn_find_mode_by_levy(alpha, min_bound=0.00001, max_bound=None, debug=False) -> float:
    assert 0 <= alpha < 1.0, f"ERROR: alpha={alpha} is not in [0, 1)"
    if 0.0 <= alpha <= 0.5:
        return 0.0
    
    if alpha > 0.5 and alpha < 0.501:
        beta = wright_m_fn_find_mode_beta_near_half(do_calc=False)
        return beta * (alpha - 0.5)
    
    if max_bound is None:
        if alpha < 0.9:
            max_bound = 2.0
        elif alpha < 0.99:
            max_bound = wright_m_fn_find_x_by_asymp_gg(1e-3, alpha)  # need to lower the max bound
            min_bound = 0.99
        else:
            max_bound = wright_m_fn_find_x_by_asymp_gg(1e-1, alpha)  # need to lower the max bound
            min_bound = 0.99

    if debug:
        print(f"DEBUG: alpha={alpha}, min_bound={min_bound}, max_bound={max_bound}")

    def _elasticity(x):
        return wright_m_fn_elasticity_by_levy(x, alpha)

    # Choose a bracket [a, b] where the function changes sign
    result = root_scalar(_elasticity, bracket=[min_bound, max_bound], method='brentq')
    if result.converged:
        return result.root
    else:
        return np.nan


def wright_m_fn_find_mode_beta_near_half(do_calc=True):
    if not do_calc:
        return 7.083591971607525  # pre-calculated value

    alpha = np.linspace(0.5001, 0.5009, 20)
    modes = np.array([wright_m_fn_find_mode_by_levy(a) for a in alpha])
    beta = np.sum(modes * (alpha - 0.5)) / np.sum((alpha - 0.5)**2)
    return beta  


def wright_m_fn_by_levy_find_x_by_step(target, alpha, step: Optional[float] = None, x_start: Optional[float] = None):
    # x must be on the right side of the mean
    # this is used primarily as a backend discovery tool for small alpha < 0.1
    # This routine is slow
    assert target >= 1e-8  # too noisy when target is too small
    if step is None:
        if alpha < 0.7:
            step = 0.05
        elif alpha < 0.9:
            step = 0.005
        else:
            step = (1-alpha) / 30.0
        
    if x_start is not None:
        x = x_start
    else:
        x = wright_m_fn_moment(alpha, 1)
    
    assert x >= 0.0
    assert step is not None and step > 0.0
    max_bound = 100.0
    while x <= max_bound:
        y = wright_m_fn_by_levy(x, alpha)
        if y <= target:
            return x
        x += step

    return np.nan  # not found


# --------------------------------------------------------------------------------
def wright_m_fn_elasticity_by_levy(x, alpha, d_log_x=0.00001):
    assert 0 <= alpha < 1, "alpha must be in [0, 1)"
    fn = lambda t: wright_m_fn_by_levy(t, alpha)
    return calc_elasticity(fn, x, d_log_x=d_log_x)


def wright_m_fn_elasticity_by_series(x, alpha, d_log_x=0.00001):
    assert 0 < alpha < 1, "alpha must be in (0, 1)"
    fn = lambda t: mainardi_wright_fn(t, alpha)
    return calc_elasticity(fn, x, d_log_x=d_log_x)




# --------------------------------------------------------------------------------
def wright_q_fn(x, alpha, max_n=40):
    return wright_ratio_fn(-x, -alpha, 0.0, delta=-1.0, max_n=max_n) * -1.0  # type: ignore


def wright_f_fn_elasticity_by_levy(x, alpha, d_log_x=0.00001):
    assert 0 < alpha < 1, "alpha must be in (0, 1)"
    fn = lambda t: wright_f_fn_by_levy(t, alpha)
    return calc_elasticity(fn, x, d_log_x=d_log_x)


class Wright_M_Elasticity:
    # calculate elasticity of M-Wright function using flat series expansion
    def __init__(self, alpha: float, min_terms: int=10, c_tol: float=1e-6):
        assert 0 < alpha < 1.0
        self.alpha = alpha
        self.min_terms = min_terms
        self.c_tol = c_tol
        self.n_terms = self.get__n_terms()
    
    @lru_cache(maxsize=1000)
    def _b(self, n):
        assert n >= 1
        return gamma(1-self.alpha) / gamma(1 - self.alpha*(n+1))


    @lru_cache(maxsize=1000)
    def _c(self, k):
        assert k >= 1
        c1 = (-1)**k / gamma(k) * self._b(k)
        c2 = sum((-1)**(j+1) / gamma(j+1) * self._b(j) * self._c(k-j) for j in range(1, k))
        return c1 + c2

    def get__n_terms(self):
        n_terms = int(self.min_terms)
        while True:
            c_last = self._c(n_terms)
            if abs(c_last) < self.c_tol:
                break
            n_terms += 1
        return n_terms

    def calc_elasticity(self, x: float):
        assert isinstance(x, float), f"ERROR: x={x} is not float"
        return sum(self._c(k) * x**k for k in range(1, self.n_terms+1))


def wright_m_fn_elasticity(x: Union[float, int, List, np.ndarray, pd.Series], alpha, min_terms=10, c_tol=1e-6):
    m = Wright_M_Elasticity(alpha, min_terms=min_terms, c_tol=c_tol)
    if isinstance(x, (list, np.ndarray, pd.Series)):
        rs = np.array([m.calc_elasticity(x1) for x1 in x])  # type: ignore
        return make_list_type(rs, x)
    
    if isinstance(x, int):  x = 1.0 * x
    assert isinstance(x, float), f"ERROR: x={x} is not float"
    return m.calc_elasticity(x)


# -------------------------------------------------------------------------------- 
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


