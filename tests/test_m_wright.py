
# test gsc and sc

import numpy as np
import pandas as pd
from numba import njit  # this causes an error: DeprecationWarning: `np.MachAr` is deprecated (NumPy 1.22).

from scipy.special import gamma
from scipy.integrate import quad
from scipy.stats import norm, levy_stable

from gas_impl.stable_count_dist import stable_count, wright_f_fn_by_sc, gsc_q_by_f
from gas_impl.stable_count_dist import levy_stable_one_sided
from gas_impl.wright import wright_fn, wright_f_fn, wright_q_fn, mainardi_wright_fn, mainardi_wright_fn_slope
from gas_impl.unit_test_utils import *


# ----------------------------------------------------------------
def test_one_sided_vs_wright():
    alpha = 0.35
    levy1 = levy_stable_one_sided(alpha)
    x = 0.85
    p = levy1.pdf(x)

    q = wright_fn(-x**(-alpha), -alpha, 0) / x
    delta_precise_up_to(p, q)

    q2 = wright_f_fn(x**(-alpha), alpha) / x
    delta_precise_up_to(p, q2)

    q3 = mainardi_wright_fn(x**(-alpha), alpha) * alpha * x**(-alpha-1)
    delta_precise_up_to(p, q3)


# ----------------------------------------------------------------
# M-Wright, supposed to be very easy to converge

def test_m_wright_vs_exp():
    x = 0.85
    p = mainardi_wright_fn(x, 0.0) 
    q = np.exp(-x)
    delta_precise_up_to(p, q)


def test_m_wright_vs_norm():
    x = 0.85
    p = mainardi_wright_fn(x, 0.5) 
    q = np.exp(-x**2/4) / np.sqrt(np.pi)
    delta_precise_up_to(p, q)

    q2 = norm(scale=np.sqrt(2)).pdf(x) * 2  # half-normal
    delta_precise_up_to(p, q2)


def test_m_wright_vs_levy2_0():
    # (A37) of Mainardi (2020)
    levy2 = levy_stable(alpha=2.0, beta=0)
    x = 0.85
    p = levy2.pdf(x)

    q = mainardi_wright_fn(x, 0.5) / 2.0
    delta_precise_up_to(p, q)

    q2 = norm(scale=np.sqrt(2)).pdf(x)  # normal, variance = 2
    delta_precise_up_to(p, q2)


def test_m_wright_moments():
    alpha = 0.47
    for n in [0.0, 1.0, 2.0]:
        def fn(x): return x**n * mainardi_wright_fn(x, alpha, max_n=80)  # max_n depends on max(x)
    
        p = quad(fn, a=0, b=8.0, limit=100000)[0]  # b can not be infinitely large, unfortunately
        q = gamma(n+1) / gamma(n*alpha + 1)
        delta_precise_up_to(p, q, abstol=0.001, reltol=0.001)


def test_m_wright_diff():
    alpha = 0.48
    z = 0.5
    dz = z * 0.0001

    f = wright_f_fn_by_sc(z, alpha)
    f_dz =  wright_f_fn_by_sc(z+dz, alpha)
    df_dz = (f_dz - f)/dz

    m = mainardi_wright_fn(z, alpha)
    m_dz = mainardi_wright_fn(z+dz, alpha)
    dm_dz = (m_dz - m)/dz
    wr = mainardi_wright_fn_slope(z, alpha)
    delta_precise_up_to(dm_dz, wr)

    df_dz2 = alpha * m + alpha * z * dm_dz
    delta_precise_up_to(df_dz, df_dz2)


def test_mu_wright_ratio():
    # this is the RV lemma
    alpha = 0.48
    z = 0.5

    f = wright_f_fn_by_sc(z, alpha)
    wr = wright_fn(-z, -alpha, -1.0)
    p1 = -wr/f 

    dz = z * 0.0001
    f_dz =  wright_f_fn_by_sc(z+dz, alpha)
    p2 = alpha * z * (f_dz - f)/dz / f + 1

    m = mainardi_wright_fn(z, alpha)
    m_dz = mainardi_wright_fn(z+dz, alpha)
    p3 = alpha * z * (m_dz - m)/dz / m + (alpha + 1)

    delta_precise_up_to(p1, p2)
    delta_precise_up_to(p2, p3)

def test_mu_wright_ratio_by_q():
    # this is the RV lemma, by Q
    alpha = 0.48
    z = 0.45

    p1 = gsc_q_by_f(z, dz_ratio=0.001, alpha=alpha)
    p2 = wright_q_fn(z, alpha)
    delta_precise_up_to(p1, p2)
    
    
def test_m_wright_slope_at_zero():
    for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
        p1 = mainardi_wright_fn_slope(0.0, alpha)
        p2 = -1/np.pi * gamma(2*alpha) * np.sin(2*alpha*np.pi) 
        delta_precise_up_to(p1, p2, msg_prefix=f"alpha = {alpha}: ")

        dz = 0.0001
        m = mainardi_wright_fn(0.0, alpha)
        m_dz = mainardi_wright_fn(dz, alpha)
        p3 = (m_dz - m)/dz
        delta_precise_up_to(p1, p3, abstol=0.01, reltol=0.001, msg_prefix=f"dm_dz alpha = {alpha}: ")