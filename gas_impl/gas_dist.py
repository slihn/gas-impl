from math import isnan
import numpy as np 
import pandas as pd
import mpmath as mp
from typing import Union
from functools import lru_cache
from scipy.stats import rv_continuous, levy_stable
from scipy.special import gamma, erf, hyp2f1, hyp1f1, betainc
from scipy.stats import norm
from scipy.integrate import quad, IntegrationWarning
from scipy.optimize import root_scalar
import cmath
import warnings

from .wright import mp_gamma
from .fcm_dist import frac_chi_mean



# --------------------------------------------------------------------------------
def gsas_moment(n: float, alpha: float, k: float) -> float:
    assert float(int(n)) == float(n)
    if n % 2 != 0: return 0.0  # odd moments are zero
    assert k != 0

    n = mp.mpf(n)
    k = mp.mpf(k)
    alpha = mp.mpf(alpha)
    # we use mpf for cancellation, not for output, but you need to set the precision
    k_sigma = mp.power(abs(k), 1/mp.mpf(2)-1/alpha)  # without the /sqrt(2) thing

    if k > 0:
        a = mp_gamma((n+1)/2) / mp.sqrt(mp.pi)
        b = mp.power(2.0 / k_sigma, n) 
        c = mp_gamma((k-1)/2) / mp_gamma((k-1)/alpha) if k != 1.0 else 2/alpha
        d = mp_gamma((k-n-1)/alpha) / mp_gamma((k-n-1)/2) if k != n+1 else alpha/2
        return float(a * b * c * d)
    if k < 0:
        a = mp_gamma((n+1)/2) / mp.sqrt(mp.pi)
        b = mp.power(k_sigma, n) 
        c = mp_gamma(abs(k)/2) / mp_gamma(abs(k)/alpha) if k != 0 else 2/alpha
        d = mp_gamma((abs(k)+n)/alpha) / mp_gamma((abs(k)+n)/2) if abs(k) != -n else alpha/2
        return float(a * b * c * d)
    raise Exception(f"ERROR: k is not handled properly")


def gsas_kurtosis(alpha: float, k:float, fisher: bool=True, exact_form=False):
    # TODO should have an analytic piece too
    if not isinstance(alpha, float):
        ans = [gsas_kurtosis(alpha1, k, fisher=fisher, exact_form=exact_form) for alpha1 in alpha]
        return ans[0] if len(ans) == 1 else ans

    k = float(k)
    if exact_form == True:
        s = 1.0/alpha
        a = 3.0 * (k-5)/(k-3)
        b = gamma(s*(k-5)) * gamma(s*(k-1)) / gamma(s*(k-3))**2
        kurt = a * b 
    else:
        kurt = gsas_moment(n=4.0, alpha=alpha, k=k) / gsas_moment(n=2.0, alpha=alpha, k=k)**2

    return (kurt - 3.0) if fisher else kurt


def gsas_pdf_at_zero(alpha, k):
    alpha = float(alpha)
    k = float(k)
    assert k != 0 
    sigma = np.power(abs(k), 1/2.0-1/alpha) / np.sqrt(2)

    if k > 0:
        g_const = gamma((k-1)/2) / gamma((k-1)/alpha) if k != 1 else 2.0 / alpha
        return sigma / np.sqrt(2*np.pi) * g_const * gamma(k/alpha) / gamma(k/2)
    if k < 0:
        g_const = gamma((abs(k)-1)/alpha) / gamma((abs(k)-1)/2) if k != -1 else alpha / 2.0
        return 1/sigma / np.sqrt(2*np.pi) * g_const * gamma(abs(k)/2) / gamma(abs(k)/alpha)


def gsas_std_pdf_at_zero(alpha, k):
    p = gsas_pdf_at_zero(alpha, k)
    var = gsas_moment(n=2.0, alpha=alpha, k=k)
    sd = var**0.5 if var >= 0 else np.NaN
    return p * sd


def gsas_characteristic_fn(x, alpha, k):
    fcm = frac_chi_mean(alpha, k)
    def fn(s):
        return norm().pdf(x/s) *  fcm.pdf(s) 

    p2 = quad(fn, a=0, b=np.inf, limit=1000)[0]
    return p2 * np.sqrt(2*np.pi)


# --------------------------------------------------------------------------------
def from_s1_to_feller(alpha, beta):  # return theta
    # Appendix A.2 of Lihn(2020)
    alpha2 = alpha * np.pi/2
    theta = cmath.phase( 1.0 -1j * beta * np.tan(alpha2)) * 2/np.pi
    assert abs(theta) <= alpha
    scale = abs( 1.0 -1j * beta * np.tan(alpha2))**(-1/alpha)
    return theta, scale


def from_feller_to_s1(alpha, theta):  # return beta and scale
    if theta == 0: return 0.0, 1.0

    alpha2 = alpha * np.pi/2
    assert abs(theta) <= alpha and abs(theta) <= 2.0-alpha

    def find_beta(beta):
        return cmath.phase( 1.0 -1j * beta * np.tan(alpha2)) * 2/np.pi - theta

    if abs(theta) == alpha:
        beta = np.sign(theta) * 1.0
    else:
        sol = root_scalar(find_beta, x0=0.5 * np.sign(theta), bracket=[-1.0, 1.0])
        beta = sol.root

    scale = abs( 1.0 -1j * beta * np.tan(alpha2))**(-1/alpha)
    return beta, scale


def levy_stable_from_feller(alpha, theta):
    beta, scale = from_feller_to_s1(alpha, theta)
    return levy_stable(alpha=alpha, beta=beta, scale=scale)


# --------------------------------------------------------------------------------
class gsas_gen(rv_continuous):

    def frac_chi_mean(self, alpha, k): return frac_chi_mean(alpha, k)

    def _pdf(self, x, alpha, k, *args, **kwargs):
        # handle array form
        if not isinstance(alpha, float):
            assert len(alpha) == len(x), f"ERROR: len of alpha and x"
            if len(x) == 1:  # trvial case
                return self._pdf(x[0], alpha=alpha[0], k=k[0])
            
            df = pd.DataFrame(data={'x': x, 'alpha': alpha, 'k': k})
            df['pdf'] = df.parallel_apply(lambda row: self._pdf(row['x'], alpha=row['alpha'], k=row['k']), axis=1)  # type: ignore
            return df['pdf'].tolist()

        # integral form
        assert isinstance(x, float)
        assert isinstance(alpha, float)
        assert isinstance(k, float)
        assert alpha >= 0
        
        def _kernel(s: float):
            return s * norm().pdf(s*x) * self.frac_chi_mean(alpha=alpha, k=k).pdf(s)

        return quad(_kernel, a=0.0, b=np.inf, limit=10000)[0]

    def _cdf(self, x, alpha, k, *args, **kwargs):
        # handle array form
        if not isinstance(alpha, float):
            assert len(alpha) == len(x), f"ERROR: len of alpha and x"
            if len(x) == 1:  # trvial case
                return self._cdf(x[0], alpha=alpha[0], k=k[0])
            
            df = pd.DataFrame(data={'x': x, 'alpha': alpha, 'k': k})
            df['cdf'] = df.parallel_apply(lambda row: self._cdf(row['x'], alpha=row['alpha'], k=row['k']), axis=1)  # type: ignore
            return df['cdf'].tolist()

        # integral form
        assert isinstance(x, float)
        assert isinstance(alpha, float)
        assert isinstance(k, float)

        def _kernel(s: float):
            return erf(s*x/np.sqrt(2)) * self.frac_chi_mean(alpha=alpha, k=k).pdf(s)

        cdf1 = quad(_kernel, a=0.0, b=np.inf, limit=10000)[0]
        return cdf1*0.5 + 0.5

    def _argcheck(self, *args, **kwargs):
        # Customize the argument checking here
        alpha = args[0]
        k = args[1]
        return (
            alpha >= 0  # Allow alpha to be zero or positive
            and k != 0 
        )

    def _munp(self, n, alpha, k, *args, **kwargs):
        # https://github.com/scipy/scipy/issues/13582
        # mu = self._munp(1, *goodargs)
        return gsas_moment(n=float(n), alpha=float(alpha), k=float(k))


gsas = gsas_gen(name="generalized symmetric alpha-stable", shapes="alpha, k")


# --------------------------------------------------------------------------------
def frac_hyp_fn(x, alpha, k, b, c):
    assert isinstance(x, float)
    fc = frac_chi_mean(alpha=alpha, k=k)

    def _kernel(s: float):
        m = hyp1f1(b, c, x*k* s**2/2) 
        return s * m * fc.pdf(s)

    fm = quad(_kernel, a=0.0, b=np.inf, limit=10000)[0]
    return fm * np.sqrt(k / (2 * np.pi))


def t_cdf_by_hyp2f1(x, k):
    k = float(k)
    x = float(x)
    c = gamma((k+1)/2) / gamma(k/2) / np.sqrt(k*np.pi)
    f21 = hyp2f1(0.5, (k+1)/2, 1.5, -x**2/k)
    cdf_integral = 2*x * c * f21
    return 0.5 + 0.5 * cdf_integral


def t_cdf_by_betainc(x, k, use_variant=False):
    k = float(k)
    if x < 0: return 1.0 - t_cdf_by_betainc(-x, k, use_variant=use_variant)
    p = k / (x**2 + k)
    cdf_integral = (1.0 - betainc(k/2, 0.5, p)) if not use_variant else betainc(0.5, k/2, 1.0-p)
    return 0.5 + 0.5 * cdf_integral


def t_cdf_by_binom(x, k):
    k = float(k)
    if x < 0: return 1.0 - t_cdf_by_binom(-x, k)
    p = k / (x**2 + k)
    n = (k-1) / 2
    ks = k/2 - 1.0
    cdf_integral = betainc(n-ks, ks+1, 1.0-p)  # should be binom(n,p).cdf(ks), but this doesn't work since we can't make both n and ks to be integers
    assert not np.isnan(cdf_integral)
    return 0.5 + 0.5 * cdf_integral


# --------------------------------------------------------------------------------
# stand-alone so that we can test it separately
def g_skew_v2_integrand(x, s, t, alpha, theta):
    q = np.cos(theta*np.pi/2)**(1/alpha)
    tau = np.tan(theta*np.pi/2)
    x1 = x*s / q
    b1 = tau * (s*t)**alpha
    return np.cos(b1 + x1*t) * np.exp(-t**2/2) / (np.pi * q)


def h_skew_v2_integrand(x, s, t, alpha, theta):
    q = np.cos(theta*np.pi/2)**(1/alpha)
    tau = np.tan(theta*np.pi/2)
    x1 = x*s / q
    b1 = tau * (s*t)**alpha
    return (np.cos(b1 + x1*t) - np.cos(x1*t)) * np.exp(-t**2/2) / (np.pi * q)


def g_skew_v2_phase(x, s, t, alpha, theta):
    q = np.cos(theta*np.pi/2)**(1/alpha)
    tau = np.tan(theta*np.pi/2)
    x1 = x*s / q
    b1 = tau * (s*t)**alpha
    return b1 + x1*t


def g_skew_v2_at_s0(alpha, theta):
    q = np.cos(theta*np.pi/2)**(1/alpha)
    g0 = 1/q * 1/np.sqrt(2*np.pi)
    return g0


def g_skew_v2_omega(x, s, alpha, theta):
    q = np.cos(theta*np.pi/2)**(1/alpha)
    tau = np.tan(theta*np.pi/2)

    w1 = x * s / q
    w2 = tau * s**alpha * (alpha if alpha > 1.0 else 1.0)
    omega = max(w1, w2)
    # print(f"w1 = {w1}, w2 = {w2} -> omega = {omega}")  # debug
    return omega

def g_skew_v2(x, s, alpha, theta, use_short_cut=True, use_t_max=True, use_adaptive=True, use_osc=True):
    # the options are mainly for testing purpose, and it is much faster
    if theta == 0 and use_short_cut:
        y = x * s
        return 1.0/np.sqrt(2*np.pi) * np.exp(-y**2/2)

    q = np.cos(theta*np.pi/2)**(1/alpha)
    tau = np.tan(theta*np.pi/2)
    omega = g_skew_v2_omega(x, s, alpha, theta)
    x1 = x*s / q

    if alpha == 1.0 and use_short_cut:
        assert abs(q) > 0.0
        y = (tau + x/q) * s
        return 1.0/q/np.sqrt(2*np.pi) * np.exp(-y**2/2)


    def integrate_osc():  
        with mp.workdps(15):  # need to control it at lower precision for speed (if it is changed inadvertantly elsewhere)
            return g_skew_v2_osc(x, s, alpha, theta, use_short_cut=False)

    if use_osc:
        if omega > 1000.0:
            return integrate_osc()

    t_max = np.inf 
    if use_t_max:  # naive b=np.inf is bad for large s, large x, hard to converge
        m = np.maximum(abs(x), abs(tau))
        t_max = 100.0 / m if m > 0.01 else np.inf
        t_max = 100.0 if t_max < 100.0 else t_max

    arg = lambda t: (tau * (s*t)**alpha + x1*t) / np.pi
    subintervals = 100000  # it needs A LOT of divisions to converge! this seems to have a limited utility once it is large enough
    def integrand(t):  return g_skew_v2_integrand(x, s, t, alpha=alpha, theta=theta)

    # def _quad(a,b): return quad(fn1, a=a, b=b, limit=subintervals)[0] / (np.pi * q)  # it needs A LOT of divisions to converge!
    def _quad(a,b): return quad(integrand, a=a, b=b, limit=subintervals)[0]  

    def integrate(a, b):
        p = None
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=UserWarning)
            try:
                p = _quad(a,b) 
            except IntegrationWarning as e:
                print(f"g_skew_v2 Warning: a,b=({a}, {b}) :: omega={omega:.2f} :: x={x}, s ={s}, alpha={alpha}, theta={theta} :: arg(a) {arg(a)} :: {type(e)} {str(e)}")
        
        if p is None: p = _quad(a,b)
        return p

    # ----------------------------------------
    # adaptive slice integral into small pieces
    if not use_adaptive:
        return integrate(a=0, b=t_max)
    
    def phase(t): return abs(g_skew_v2_phase(x, s, t, alpha=alpha, theta=theta))
    phase_max = subintervals/1000.0 * np.pi
 
    t_max = np.minimum(t_max, 100.0)  # type: ignore  # make sure it is a number, and not crazily large
    if phase(t_max) <= phase_max:
        return integrate(a=0, b=t_max)

    # ----------------------------------------
    #  get the first interval
    # TODO this logic is okay for smaller abs(theta), but still not good enough for larger abs(theta)
    t1 = 0.0
    t2 = t_max
   
    def round_phase(t2, iter, t1) -> float:
        def find_t2(ph) -> float:
            def fn(t): return abs(phase(t)) - (ph * iter + 0.5 * np.pi)
            t2n = root_scalar(fn, x0=t2, x1=t2+10.0, maxiter=1000).root
            return t2n

        retry = 1.0
        while retry < 10:
            t2n = find_t2(phase_max/retry)
            if t2n > t1: break;
            retry = retry + 1
        return np.nan

    while phase(t2) > phase_max:
        t2 = t2 / 2.0

    iter = 1.0
    t2 = round_phase(t2, iter, t1)
    if not (t2 > t1) or np.isnan(t2):
        return integrate_osc()  # forget about quad, let's do osc
    assert t1 < t2, f"ERROR: t1 < t2 failed at iter {iter}, {t1} vs {t2} :: x={x}, alpha={alpha}, theta={theta}"
    p = integrate(t1, t2)
    # print([t1, t2, p])  # smallest t for [0, t]

    # iterate through next intevals
    iter = 2.0
    t1 = t2
    while t1 < t_max:
        t2 = t1 * 1.25
        while phase(t2) - phase(t1) < phase_max:
            t2 = t2 * 1.25

        t2 = round_phase(t2, iter, t1)
        if not (t2 > t1) or np.isnan(t2):
            return integrate_osc()  # forget about quad, let's do osc
        assert t1 < t2, f"ERROR: t1 < t2 failed at iter {iter}, {t1} vs {t2} :: x={x}, alpha={alpha}, theta={theta}"
        p1 = integrate(t1, t2)
        if abs(p) > 0:
            contrib = abs(p1/p)  # reltol
        else:
            contrib = p1

        p = p + p1
        # print([t1, t2, p, contrib])
        t1 = t2
        iter = iter + 1
        if abs(p) > 0 and contrib < 1e-8:
            break

    return p


# use mp.quadosc to handle high oscilatory function
# it is much faster than quad-based algo when omega > 200
# and become absolutely necessary when omega > 10000
# but it can take longer than 1s when omega is small, it is not good at a relatively flat function
def g_skew_v2_osc(x, s, alpha, theta, use_short_cut=True):
    # the options are mainly for testing purpose, and it is much faster
    if theta == 0 and use_short_cut:
        y = x * s
        return 1.0/np.sqrt(2*np.pi) * np.exp(-y**2/2)

    q = mp.cos(theta*mp.pi/2)**(1/alpha)
    tau = mp.tan(theta*mp.pi/2)
 
    if alpha == 1.0 and use_short_cut:
        assert abs(q) > 0.0
        y = float((tau + x/q) * s)
        return 1.0/q/np.sqrt(2*np.pi) * np.exp(-y**2/2)

    assert abs(theta) <= min(alpha, 2.0-alpha)
    assert abs(theta) != 1.0
    # tau is infinite, q is zero, when theta = 1; theta can only be 1 when alpha = 1
    # which should produce L_1^1(x) = delta(x+1)

    def _v2_integrand(t):
        x1 = x * s / q
        b1 = tau * (s*t)**alpha
        return mp.cos(b1 + x1*t) * mp.exp(-t**2/2) / (mp.pi * q)

    omega = g_skew_v2_omega(x, s, alpha, theta)
    p = mp.quadosc(_v2_integrand, [0, mp.inf], omega=omega)
    return float(mp.re(p))


# --------------------------------------------------------------------------------
class lihn_stable_gen(rv_continuous):

    def frac_chi_mean(self, alpha, k): return frac_chi_mean(alpha, k)

    def g_skew_v2(self, x, s, alpha, theta, use_short_cut=True, use_t_max=True):
        return g_skew_v2(x, s, alpha, theta, use_short_cut=use_short_cut, use_t_max=use_t_max)

    def _pdf(self, x, alpha, k, theta, *args, **kwargs):
        # handle array form
        if not isinstance(alpha, float):
            assert len(alpha) == len(x), f"ERROR: len of alpha and x"
            if len(x) == 1:  # trvial case
                return self._pdf(x[0], alpha=alpha[0], k=k[0], theta=theta[0])
            
            df = pd.DataFrame(data={'x': x, 'alpha': alpha, 'k': k, 'theta': theta})
            df['pdf'] = df.parallel_apply(lambda row: self._pdf(row['x'], alpha=row['alpha'], k=row['k'], theta=row['theta']), axis=1)  # type: ignore
            return df['pdf'].tolist()
            # return [self._pdf(x1, df=df1, alpha=a1) for x1, df1, a1 in zip(x, df, alpha)]

        # integral form
        assert isinstance(x, float)
        assert isinstance(alpha, float)
        assert isinstance(k, float)

        def _kernel(s: float):
            return s * self.g_skew_v2(x, s, alpha, theta) * self.frac_chi_mean(alpha=alpha, k=k).pdf(s)

        p = quad(_kernel, a=0.0, b=np.inf, limit=10000)[0]
        return p

    def _argcheck(self, *args, **kwargs):
        # Customize the argument checking here
        alpha = args[0]
        k = args[1]
        theta = args[2]
        return (
            alpha >= 0  # Allow alpha to be zero or positive
            and k != 0
            and abs(theta) <= np.minimum(alpha, 2.0-alpha) # F.35 of Appendix F
        )


lihn_stable = lihn_stable_gen(name="generalized alpha-stable", shapes="alpha, k, theta")
# theta is Feller's parametrization

