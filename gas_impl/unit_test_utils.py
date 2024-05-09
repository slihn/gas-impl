import numpy as np
import pandas as pd
from scipy.special import gamma
from scipy.integrate import quad
from scipy import stats
from scipy.stats import gengamma


def delta_precise_up_to(p, q, abstol=1e-4, reltol=1e-4, msg_prefix=""):
    actual1 = abs(p - q)
    actual2 = abs(q/p - 1.0) if p*q != 0 else np.nan
    assert actual1 < abstol, f"ERROR: {msg_prefix} abstol failed: x={p}, y={q}, actual {actual1} vs {abstol}"
    if p*q != 0:
        assert actual2 < reltol, f"ERROR: {msg_prefix} reltol failed: x={p}, y={q}, actual {actual2} vs {reltol}"



def compare_two_rvs(x, rv1, rv2, min_p=0.1, abstol=1e-4, reltol=1e-4, msg_prefix="") -> float:  # return p1 for further usage
    p1 = rv1.pdf(x)
    p2 = rv2.pdf(x)
    assert p1 > min_p, f"ERROR: {msg_prefix} p1 = {p1} is not greater than min_p = {min_p}"
    delta_precise_up_to(p1, p2, abstol=abstol, reltol=reltol, msg_prefix=msg_prefix)
    return p1


def compare_cdf_of_two_rvs(x, rv1, rv2, abstol=1e-4, reltol=1e-4, msg_prefix="") -> float:  # return p1 for further usage
    p1 = rv1.cdf(x)
    p2 = rv2.cdf(x)
    assert p1 >= 0, f"ERROR: {msg_prefix} p1 = {p1} is not greater than or equal to 0"
    delta_precise_up_to(p1, p2, abstol=abstol, reltol=reltol, msg_prefix=msg_prefix)
    return p1


def product_dist_test_suite(p1, x, unit_fn, gsc):
    def fn1(s):
        return 1.0 / s * unit_fn(x/s) *  gsc.pdf(s)

    p1i = quad(fn1, a=0, b=np.inf, limit=1000)[0]
    delta_precise_up_to(p1, p1i)


def ratio_dist_test_suite(p1, x, unit_fn, gsc):
    def fn1(s):
        return s * unit_fn(s*x) *  gsc.pdf(s)

    p1i = quad(fn1, a=0, b=np.inf, limit=1000)[0]
    delta_precise_up_to(p1, p1i)


# ---------------------------------------------------------------

def stretched1(x, alpha):
    return 1.0 / gamma(1/alpha+1) * np.exp(-x**alpha)

def laplace1(x):
    return np.exp(-x)

def laplace(x):
    return np.exp(-abs(x)) / 2.0  # this is Wikipedia's Laplace distribution

def pdf_gg(x, a, d, p):
    c = p/a / gamma(d/p) 
    return c * (x/a)**(d-1) * np.exp(-(x/a)**p)

def gg_rv(a, d, p):
    return gengamma(a=d/p, c=p, scale=a)
