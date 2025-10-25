# this file contains the best wright_m_fn and wright_f_fn 
# that combines the levy, series, and asymptotic approximations
# and the functions work for x from 0 to as large as float64 can represent
# and for alpha from 0.001 to 0.999 (inclusive)

from functools import lru_cache
import numpy as np
import pandas as pd
from typing import List
from numpy import polyval, polyfit  # type: ignore

from .wright_asymp import wright_m_fn_moment, wright_m_fn_find_x_by_asymp_gg, wright_m_fn_asymp_paris
from .wright import wright_m_fn_by_levy, wright_mainardi_fn_ts, wright_m_fn_find_x_by_asymp_gg, wright_m_fn_by_levy_find_x_by_step
from .utils import make_list_type



def wright_f_fn_by_levy_asymp(x, alpha, use_coef_hardcoded=True):
    # this power function can be used for fractional gamma and FCM
    assert isinstance(alpha, float), "alpha should be a float"

    def _calc1(xi):
        assert isinstance(xi, float), "xi should be a float"
        y = wright_m_fn_by_levy_asymp(xi, alpha, use_coef_hardcoded=use_coef_hardcoded)
        assert isinstance(y, float), "y should be a float"
        return y * xi * alpha

    if isinstance(x, (list, np.ndarray, pd.Series)):
        rs = np.array([_calc1(xi) for xi in x])  # type: ignore
        return make_list_type(rs, x)
    else:
        return _calc1(x)


def wright_m_fn_by_levy_asymp(x, alpha, use_coef_hardcoded=True):
    # split this into three regions:
    # small: alpha < 0.1
    # large: alpha > 0.9
    # middle: 0.1 <= alpha <= 0.9
    def _calc1(xi: float) -> float:
        assert isinstance(xi, float), "xi should be a float"
        if 0.1 <= alpha <= 0.9:
            return wright_m_fn_by_levy_asymp_aux_middle(xi, alpha, use_coef_hardcoded)
        elif alpha < 0.1:
            return wright_m_fn_by_levy_asymp_aux_small(xi, alpha, use_coef_hardcoded=False)  # type: ignore
        elif alpha > 0.9 and alpha <= 0.993:
            return wright_m_fn_by_levy_asymp_aux_large(xi, alpha)  # type: ignore
        elif alpha > 0.993 and alpha <= 0.998:
            # more to be implemented, especially issues when alpha > 0.994
            return wright_m_fn_by_levy_asymp_aux_ultra_large(xi, alpha)  # type: ignore
        elif alpha == 1.0:
            return wright_m_fn_by_levy(xi, alpha)  # type: ignore
        else:
            raise ValueError("alpha should be between 0 and 1")

    if isinstance(x, (list, np.ndarray, pd.Series)):
        rs = np.array([_calc1(xi) for xi in x])  # type: ignore
        return make_list_type(rs, x)
    else:
        return _calc1(x)


# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
@lru_cache(maxsize=128)
def get_poly_fit_aux_small(alpha):
    x5 = wright_m_fn_by_levy_find_x_by_step(1e-5, alpha)
    x6 = wright_m_fn_by_levy_find_x_by_step(1e-6, alpha)
    # this is very slow
    x_5_6 = np.linspace(x5, x6, 200)
    xk = x_5_6**(1.0/(1.0-alpha))
    log_f = np.log(wright_m_fn_by_levy(x_5_6, alpha))  # type: ignore
    poly = polyfit(xk, log_f, 1)
    return x5, x6, poly


def wright_m_fn_by_levy_asymp_aux_small(x, alpha, use_coef_hardcoded=False) -> float:
    # for middle region, we interpolate between levy and paris asymp during function values in [1e-5, 1e-6]
    assert 0.001 <= alpha <= 0.1, "alpha should be between 0.001 and 0.1"
    assert isinstance(x, float), "x should be a float"
    assert x >= 0, "x should be non-negative"
    
    if use_coef_hardcoded:
        x5, x6 = np.nan, np.nan  # to be implemented
        poly = [np.nan, np.nan]  # to be implemented
    else:
        x5, x6, poly = get_poly_fit_aux_small(alpha)
        # this is very slow

    def _wright_m_fn_asymp_small_alpha(x): 
        log_y = polyval(poly, x**(1.0/(1.0-alpha)))
        return np.exp(log_y)

    assert x5 < x6, f"x5 should be less than x6: x5={x5}, x6={x6}"
    assert x5 > 0 and x6 > 0, "x5 and x6 should be positive"

    if x <= x5:
        return wright_m_fn_by_levy(x, alpha)  # type: ignore
    elif x >= x6:
        return _wright_m_fn_asymp_small_alpha(x)  # type: ignore
    else:
        # linear interpolation in log space
        y5 = wright_m_fn_by_levy(x, alpha)
        y6 = _wright_m_fn_asymp_small_alpha(x)
        assert isinstance(y5, float) and isinstance(y6, float)
        assert y5 > 0 and y6 > 0, "y5 and y6 should be positive"
        logy = np.log(y5) + (np.log(y6) - np.log(y5)) * (x - x5) / (x6 - x5)
        return np.exp(logy)



# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
middle_coef_x5_hardcoded = [
    -1.95469743e+01,  7.94798612e+01, -1.39662374e+02,  1.37002225e+02,
    -8.09641038e+01,  2.77829089e+01, -4.00243789e+00, -2.61696329e+00,
    9.84796429e-02,  2.46654662e+00 ]

middle_coef_x6_hardcoded = [
    -16.51331666,   64.58299919, -107.7358991,    97.77823198,  -50.40172747,
    11.96082219,    1.62188172,   -4.03660161,    0.13638296,    2.6444866 ]


@lru_cache(maxsize=128)
def _get_middle_coef_x5_x6_hardcoded(alpha):
    x5 = np.exp(polyval(middle_coef_x5_hardcoded, alpha))
    x6 = np.exp(polyval(middle_coef_x6_hardcoded, alpha))
    return x5, x6


def wright_m_fn_by_levy_asymp_aux_middle(x, alpha, use_coef_hardcoded=True) -> float:
    # for middle region, we interpolate between levy and paris asymp during function values in [1e-5, 1e-6]
    assert 0.1 <= alpha <= 0.9, "alpha should be between 0.1 and 0.9"
    assert isinstance(x, float), "x should be a float"
    assert x >= 0, "x should be non-negative"
    
    if use_coef_hardcoded:
        x5, x6 = _get_middle_coef_x5_x6_hardcoded(alpha)
    else:
        x5 = wright_m_fn_find_x_by_asymp_gg(1e-5, alpha)
        x6 = wright_m_fn_find_x_by_asymp_gg(1e-6, alpha)
    return _wright_m_fn_interpolated(x, alpha, x5, x6)


def _wright_m_fn_interpolated(x, alpha, xl, xu, use_ts=False) -> float:
    assert xl < xu, f"xl should be less than xu: xl={xl}, xu={xu}"
    assert xl > 0 and xu > 0, "xl and xu should be positive"

    fn = wright_m_fn_by_levy if not use_ts else wright_mainardi_fn_ts
    if x <= xl:
        return fn(x, alpha)  # type: ignore
    elif x >= xu:
        return wright_m_fn_asymp_paris(x, alpha)  # type: ignore
    else:
        # linear interpolation in log space
        yl = fn(x, alpha)  # type: ignore
        yu = wright_m_fn_asymp_paris(x, alpha)
        assert isinstance(yl, float) and isinstance(yu, float)
        assert yl > 0 and yu > 0, "yl and yu should be positive"
        logy = np.log(yl) + (np.log(yu) - np.log(yl)) * (x - xl) / (xu - xl)
        return np.exp(logy)


# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
large_coef_x3_lower: List[float] = [  # this is target = 5e-3
    1.28677472e-06, 5.34328662e-04, 1.23699992e-02, 1.07278095e-01,
    1.29991609e+00, 2.18450294e+00 ] 
large_coef_x3_upper: List[float] = [  # this is target = 1e-3
    2.73423544e-05, 1.23042751e-03, 2.00318556e-02, 1.51591274e-01,
    1.43961708e+00, 2.39737391e+00 ]
large_coef_x4: List[float] = [  # this is target = 1e-4
    1.22387142e-04, 3.60569580e-03, 4.36100138e-02, 2.69095639e-01,
    1.74247184e+00, 2.75514014e+00 ]  


@lru_cache(maxsize=128)
def get_x_by_large_coef_idx(alpha, coef_idx: int) -> float:
    if coef_idx == 1:
        coef = large_coef_x3_lower
    elif coef_idx == 2:
        coef = large_coef_x3_upper
    elif coef_idx == 3:
        coef = large_coef_x4
    else:
        raise Exception(f"ERROR: coef_idx {coef_idx} is unknown")
    return get_x_by_large_coef(alpha, coef)


def get_x_by_large_coef(alpha, coef: List[float]) -> float:
    assert len(coef) >= 2, "coef should be of length 2 at least"
    eps = 1.0 - alpha
    dx = float(np.exp(np.polyval(coef, np.log(eps))))
    return 1.0 + dx


def wright_m_fn_by_levy_asymp_aux_large(x, alpha) -> float:
    assert 0.9 <= alpha <= 0.993, "alpha should be between 0.9 and 0.993"
    assert isinstance(x, float), "x should be a float"
    assert x >= 0, "x should be non-negative"
    
    xl = get_x_by_large_coef_idx(alpha, 1)
    xu = get_x_by_large_coef_idx(alpha, 2)
    return _wright_m_fn_interpolated(x, alpha, xl, xu)


def wright_m_fn_by_levy_asymp_aux_ultra_large(x, alpha) -> float:
    assert 0.993 < alpha <= 0.999, "alpha should be between 0.993 and 0.999"
    assert isinstance(x, float), "x should be a float"
    assert x >= 0, "x should be non-negative"
    
    xl = get_x_by_large_coef_idx(alpha, 1)
    xu = get_x_by_large_coef_idx(alpha, 2)
    return _wright_m_fn_interpolated(x, alpha, xl, xu, use_ts=True)
