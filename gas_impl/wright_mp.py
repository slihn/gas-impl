# the mpmath version of the Wright function
# primarily to calibrate the asymptotic behavior of Wright, FG, and FCM

import mpmath as mp
from mpmath import mpf
import numpy as np
import pandas as pd
from datetime import date, datetime
from typing import Union, List, Optional

from .wright_asymp import wright_m_fn_moment
from .utils import make_list_type, calc_elasticity_mp


def wright_m_fn_mp(x, alpha, max_n = 1000):
    if isinstance(x, (list, np.ndarray, pd.Series)):
        rs = np.array([wright_m_fn_mp(x1, alpha, max_n=max_n) for x1 in x])  # type: ignore
        return make_list_type(rs, x)

    start = 1
    x = mpf(x)
    alpha = mpf(alpha)
    assert alpha >= 0 and alpha < 1.0, f"ERROR: alpha={alpha} must be in [0,1)"
    
    def wright1(n, x):
        n = mpf(n)
        g = mp.gamma(alpha * n) * mp.sin(alpha * n * mp.pi) if alpha != 0 else mp.pi
        return mp.power(-x, n-1) / mp.gamma(n) * g 

    if max_n <= 10000 and mp.mp.prec <= 1024:
        p = np.array([wright1(k, x) for k in range(start, max_n+1)])
        p = sum(p) 
    else:
        p = mpf(0)
        for k in range(start, max_n+1):  p = p + wright1(k, x)
    return p / mp.pi


def wright_f_fn_mp(x, alpha, max_n = 1000):
    if isinstance(x, (list, np.ndarray, pd.Series)):
        rs = np.array([wright_f_fn_mp(x1, alpha, max_n=max_n) for x1 in x])  # type: ignore
        return make_list_type(rs, x)

    x = mpf(x)
    alpha = mpf(alpha)
    return wright_m_fn_mp(x, alpha, max_n=max_n) * alpha * x


def wright_m_fn_mp_elasticity(x, alpha, max_n = 1000, d_log_x = mpf(0.000001), debug=False):
    assert 0 <= alpha < 1, "alpha must be in [0, 1)"
    fn = lambda t: wright_m_fn_mp(t, alpha, max_n=max_n)
    return calc_elasticity_mp(fn, x, d_log_x=d_log_x, debug=debug)


def wright_m_fn_mp_slope(x, alpha, max_n = 1000, d_log_x = mpf(0.000001)):
    assert 0 <= alpha < 1, "alpha must be in [0, 1)"
    if isinstance(x, (list, np.ndarray, pd.Series)):
        rs = np.array([wright_m_fn_mp_slope(x1, alpha, max_n=max_n, d_log_x=d_log_x) for x1 in x])  # type: ignore
        return make_list_type(rs, x)

    x1 = mpf(x)
    x2 = x1 * mp.exp(d_log_x)  # x + dx
    assert isinstance(x1, mpf) and isinstance(x2, mpf)
    m = wright_m_fn_mp(np.array([x1, x2]), alpha, max_n=max_n)
    return (m[1] - m[0]) / (x2 - x1)


# --------------------------------------------------------------------------------
# this uses tsquad to carry out Prodanov integral of the M-Wright function
# (11) and Theorem 1 of Prodanov (2023): Computation of the Wright function from its integral representation
def wright_mainardi_fn_ts_mp(x, alpha, integrand_r=None):
    # this can get to 0.998, but not 0.999, alas !
    # mp or not, doesn't make it easy to converge for small numbers
    if alpha == 0: return mp.exp(-x)
    assert alpha > 0 and alpha < 1.0
    return wright_fn_ts_mp(x, alpha, b = mpf(1.0) - mpf(alpha), integrand_r=integrand_r)


def wright_mainardi_fn_cdf_ts(x, alpha):
    # this is doing poorly when alpha is too large or too small (< 0.1), and/or x is too large (x > 100)
    # max(x) is getting smaller when alpha > 0.5, you can use mainardi_wright_fn_cdf_by_levy() instead
    # the integral at b = 1 is not stable
    assert alpha > 0 and alpha < 1.0
    return mpf(-1.0) * wright_fn_ts_mp(x, alpha, b=mpf(1.0))


# this uses tsquad to carry out Prodanov integral of the Wright function of W_{-alpha, b}(-x) type
# M-Wright is just a special case of it

def wright_fn_ts_mp(x, alpha, b, integrand_r=None):
    # if integrand_r is provided, you want to evaluate the integrand only
    assert 0 <= b <= 1
    b = mpf(b)
    a = -mpf(alpha)
    z = -mpf(x)
    sin_a = mp.sin(a * mp.pi)  # this is small when alpha is close to 1
    cos_a = mp.cos(a * mp.pi)

    def integrand(r):
        r = mpf(r)
        zra = z / r**a
        return (
            mp.sin(sin_a * zra + mp.pi * b) *
            mp.exp(cos_a * zra - r) /
            (mp.pi * r**b)
        )
    if integrand_r is not None:
        return integrand(mpf(integrand_r))

    try:
        return mp.quadts(integrand, [0, mp.inf], maxdegree=10)  # same domain as the tsquad call
    except Exception as e:
        print(f"ERROR: x={float(x)}, alpha={float(alpha)}, b={float(b)}, {e}")
        return mp.nan


# --------------------------------------------------------------------------
# the use case is to find target = 1e-3 for every alpha between 0.99 and 1.0
# so that we can calibrate the asymptotic formula for this range of alpha
# be careful about the precision requirement when alpha is very close to 1.0
# --------------------------------------------------------------------------
def wright_m_fn_mp_find_x_by_step(target, alpha, max_n = 1000, 
        step: Optional[float] = None, x_start: Optional[float] = None, debug: Optional[int] = None):
    # x must be on the right side of the mean
    # this is used primarily as a backend discovery tool for large alpha > 0.99
    # This routine is very slow
    assert target >= 1e-8  # too noisy when target is too small
    if step is None:
        if alpha < 0.7:
            step = 0.01
        elif alpha < 0.9:
            step = 0.005
        elif alpha < 0.99:
            step = 0.001
        else:
            step = (1-alpha) / 100.0
        
    if x_start is not None:
        x = x_start
    else:
        x = wright_m_fn_moment(alpha, 1)
    
    assert x >= 0.0, f"ERROR: x must be positive, x = {x}"
    assert step is not None and step > 0.0, f"ERROR: step must be positive, step = {step}"
    x = prev_x = mpf(x)
    prev_y = mp.nan
    alpha = mpf(alpha)
    step = mpf(step)
    big_step = mpf(0.01) if alpha > 0.99 else mpf(0.1)
    use_step = step

    target = mpf(target)
    max_bound = mpf(100.0)

    cnt = 0
    start_time = timer_time = datetime.now()
    elapsed_time = 0.0
    while x <= max_bound:
        y = wright_m_fn_mp(x, alpha, max_n=max_n)
        assert isinstance(y, mpf)
        if y <= target:
            if y <= mpf(0.1) * target:  # use smaller step if y drops too fast
                if use_step == big_step:
                    if y > mpf(0):
                        big_step = big_step / mpf(4.0)
                    else:
                        big_step = big_step / mpf(10.0)
                else:
                    step = step / mpf(2.0)

                if debug is not None and debug > 0:
                    now = datetime.now()
                    elapsed_time = (now - start_time).total_seconds()
                    use_step = step if prev_y < mpf(0.1) else big_step
                    print(f"RETRO: cnt={cnt}, x={float(x):.8f}, y={float(prev_y):.8f}, target={float(target):.8f}, use_step={float(use_step):.8f}, elapsed_time={elapsed_time:.1f}s")  # type: ignore

                x = prev_x
                y = prev_y
                continue
            return x

        use_step = step if y < mpf(0.1) else big_step
        cnt += 1

        if debug is not None and debug > 0:
            now = datetime.now()
            elapsed_time = (now - start_time).total_seconds()
            if (now - timer_time).total_seconds() >= debug or cnt == 1:
                print(f"DEBUG: cnt={cnt}, x={float(x):.8f}, y={float(y):.8f}, target={float(target):.8f}, use_step={float(use_step):.8f}, elapsed_time={elapsed_time:.1f}s")  # type: ignore
                timer_time = now

        prev_x = x
        prev_y = y
        x += use_step

    return mp.nan  # not found
