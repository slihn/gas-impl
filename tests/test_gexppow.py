
# test gsas

import numpy as np
import pandas as pd
from scipy.special import gamma
from scipy.integrate import quad
from scipy import stats
from scipy.stats import norm, cauchy, levy_stable, gengamma

from gas_impl.gas_dist import gsas, gsas_characteristic_fn
from gas_impl.fcm_dist import fcm_moment
from gas_impl.gexppow_dist import gexppow, gexppow_pdf_at_zero, gexppow_std_pdf_at_zero, gexppow_moment, gexppow_kurtosis
from gas_impl.unit_test_utils import *


# -------------------------------------------------------------------------------------
def _cauchy(x): return cauchy().pdf(x) 
def _norm(x): return norm().pdf(x) 

def exppow(c):
    return gengamma(a=1/c, c=c)


# -------------------------------------------------------------------------------------
# basic stuff

def test_exppow_pdf_equiv():
    x = 0.25
    for alpha in [0.75, 1.0, 1.5, 1.9]:
        p1 = exppow(alpha).pdf(x) / 2  # two sided
        p2 = gexppow(alpha=alpha, k=1.0).pdf(x)
        delta_precise_up_to(p1, p2, msg_prefix=f"alpha {alpha}")


def test_exppow_cdf_equiv():
    x = 0.25
    for alpha in [0.75, 1.0, 1.5, 1.9]:
        assert x >= 0
        p1 = exppow(alpha).cdf(x) / 2 + 0.5  # two sided
        p2 = gexppow(alpha=alpha, k=1.0).cdf(x)
        delta_precise_up_to(p1, p2, msg_prefix=f"alpha {alpha}")


# -------------------------------------------------------------------------------------
# gexppow vs gsas
def test_gexppow_cdf_equal_gsas():
    x = 0.35
    for alpha in [0.75, 1.0, 1.25]:
        for k in [3.1, 4.3, 5.2]:
            p1 = gexppow(alpha=alpha, k=k).cdf(x)
            p2 = gsas(alpha=alpha, k=-k).cdf(x)
            delta_precise_up_to(p1, p2)


def test_gexppow_pdf_equal_gsas_cf():
    x = 0.25
    for alpha in [0.75, 1.0, 1.25]:
        for k in [3.1, 4.3, 5.2]:
            p1 = gexppow(alpha=alpha, k=k).pdf(x)
            p2 = gsas_characteristic_fn(x, alpha=alpha, k=k)
            p3 = gsas(alpha=alpha, k=-k).pdf(x)
            m1 = fcm_moment(1.0, alpha=alpha, k=k) * np.sqrt(2*np.pi)
            delta_precise_up_to(p2, p1 * m1)
            delta_precise_up_to(p2, p3 * m1)


# -------------------------------------------------------------------------------------
# gexppow
def test_exppow_pdf_at_zero_equiv():
    alpha = 0.75
    g = gexppow(alpha=alpha, k=1.0)

    p1 = gexppow_pdf_at_zero(alpha=alpha, k=1.0)
    p2 = 0.5 / gamma(1/alpha+1) 
    delta_precise_up_to(p1, p2)
    
    delta_precise_up_to(p1, g.pdf(0))


def test_exppow_pdf_at_zero_equiv_norm():
    p1 = gexppow_std_pdf_at_zero(alpha=2.0, k=1.0)
    p2 = norm().pdf(0.0)
    delta_precise_up_to(p1, p2)


def test_exppow_pdf_at_zero_equiv_laplace():
    p1 = gexppow_std_pdf_at_zero(alpha=1.0, k=1.0)
    p2 = stats.laplace().pdf(0) * stats.laplace().std()
    delta_precise_up_to(p1, p2)


def test_gexppow_kurtosis():
    for alpha in [0.75, 1.0, 1.25]:
        for k in [3.1, 4.3, 5.2]:
            g = gexppow(alpha=alpha, k=k)
            p1 = g.moment(4.0) / g.moment(2.0)**2 - 3.0
            p2 = gexppow_kurtosis(alpha, k)
            delta_precise_up_to(p1, p2)


class Test_GExpPow_PDF0:
    alpha = 0.85
    k = 2.1
    g = gexppow(alpha=alpha, k=k)

    def test_pdf_at_zero(self):
        p1 = gexppow_pdf_at_zero(alpha=self.alpha, k=self.k)
        p2 = self.g.pdf(0.0)
        delta_precise_up_to(p1, p2)

    def test_std_pdf_at_zero(self):
        p1 = gexppow_std_pdf_at_zero(alpha=self.alpha, k=self.k)
        sd = gexppow_moment(n=2.0, alpha=self.alpha, k=self.k)**0.5
        p2 = gexppow(alpha=self.alpha, k=self.k, scale=1/sd).pdf(0.0)
        delta_precise_up_to(p1, p2)

    def test_cdf_at_zero(self):
        p1 = self.g.cdf(0.0)
        p2 = 0.5
        delta_precise_up_to(p1, p2)

    def test_cdf_interval(self):
        x = 0.2
        def _kernel(z): return self.g.pdf(z)
        p1 = quad(_kernel, a=-x, b=x, limit=10000)[0]  # this is slow, takes 30 seconds or more
        p2 = self.g.cdf(x) - self.g.cdf(-x)
        delta_precise_up_to(p1, p2)


# moment tests are in test_hard_gas.py
