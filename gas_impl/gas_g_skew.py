
import numpy as np 
import pandas as pd
import mpmath as mp
from typing import Union, Optional
from scipy.integrate import quad, IntegrationWarning
from scipy.optimize import root_scalar
import warnings


# --------------------------------------------------------------------------------
# s_skew function:
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
    if not (t2 > t1) or np.isnan(t2):  # type: ignore
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
        if not (t2 > t1) or np.isnan(t2):  # type: ignore
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


