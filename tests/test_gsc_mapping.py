
# test gsc and sc
# pyright: reportGeneralTypeIssues=false

import numpy as np
import pandas as pd
from numba import njit  # this causes an error: DeprecationWarning: `np.MachAr` is deprecated (NumPy 1.22).

from scipy.special import gamma
from scipy.integrate import quad
from scipy import stats
from scipy.stats import gengamma, norm, weibull_min, rayleigh, poisson, chi, chi2, invgamma, invweibull

from gas_impl.stable_count_dist import stable_count, gen_stable_count, wright_f_fn_by_sc, stable_vol
from gas_impl.wright import wright_fn, wright_f_fn, wright_m_fn, mainardi_wright_fn
from gas_impl.hankel import *
from gas_impl.unit_test_utils import *


def weibull_gsc(df, precise=True):
    alpha = 0.0 if precise else 0.1
    return gen_stable_count(alpha=alpha, sigma=1.0, d=0.0, p=df)


# ----------------------------------------------------------------
# ----------------------------------------------------------------
# ----------------------------------------------------------------
# classic

def _dual_compare(x, rv, gsc1, gsc2, reltol=0.03):
    compare_two_rvs(x, rv, gsc1, msg_prefix="gsc1")  # precise at alpha=0
    compare_two_rvs(x, rv, gsc2, abstol=0.1, reltol=reltol, msg_prefix="gsc2")  # approx at alpha=0.1


def test_gsc0_stretched_exp():
    alpha = 0.8
    stretched = gengamma(a=1.0/alpha, c=alpha)  # a = d / c, c = p
    x = stretched.moment(1)
    gsc1 = gen_stable_count(alpha=0.0, sigma=1.0, d=1.0-alpha, p=alpha)
    gsc2 = gen_stable_count(alpha=0.1, sigma=1.0, d=1.0-alpha, p=alpha)
    _dual_compare(x, stretched, gsc1, gsc2)


def test_gsc0_normal():
    x = 0.85
    p = norm().pdf(x) * 2  # half-normal
    gsc = gen_stable_count(alpha=0.0, sigma=np.sqrt(2), d=-1.0, p=2.0)
    assert p > 0.2
    delta_precise_up_to(p, gsc.pdf(x))

    gsc0 = gen_stable_count(alpha=0.1, sigma=np.sqrt(2), d=-1.0, p=2.0)
    delta_precise_up_to(p, gsc0.pdf(x), abstol=0.1, reltol=0.03)


def test_gsc0_weibull():
    k = 0.55
    wb = weibull_min(c=k)
    x = wb.moment(1)
    gsc1 = weibull_gsc(df=k)
    gsc2 = weibull_gsc(df=k, precise=False)
    _dual_compare(x, wb, gsc1, gsc2)


# laplace1 test skipped, since it is just weibull(1)

def test_gsc0_rayleigh():
    x = 0.35
    gsc1 = gen_stable_count(alpha=0.0,  sigma=np.sqrt(2), d=0.0, p=2.0)
    gsc2 = gen_stable_count(alpha=0.05, sigma=np.sqrt(2), d=0.0, p=2.0)
    _dual_compare(x, rayleigh(), gsc1, gsc2)


def test_gsc0_equal_gamma():
    a = 1.5
    gg = gengamma(a=a, c=1.0, scale=1.5)  # a = d
    x = gg.moment(1)
    gsc1 = gen_stable_count(alpha=0.0, sigma=1.5, d=a-1, p=1.0)
    gsc2 = gen_stable_count(alpha=0.1, sigma=1.5, d=a-1, p=1.0)
    _dual_compare(x, gg, gsc1, gsc2)


def test_gsc0_equal_chi():
    k = 3.0
    rv_chi = chi(df=k)
    x = rv_chi.moment(1)
    gsc1 = gen_stable_count(alpha=0.0, sigma=np.sqrt(2), d=k-2, p=2.0)
    gsc2 = gen_stable_count(alpha=0.1, sigma=np.sqrt(2), d=k-2, p=2.0)
    _dual_compare(x, rv_chi, gsc1, gsc2)


def test_gsc0_equal_chi2():
    k = 2.8
    rv_chi2 = chi2(df=k)
    x = rv_chi2.moment(1)
    gsc1 = gen_stable_count(alpha=0.0, sigma=2.0, d=k/2-1, p=1.0)
    gsc2 = gen_stable_count(alpha=0.1, sigma=2.0, d=k/2-1, p=1.0)
    _dual_compare(x, rv_chi2, gsc1, gsc2)


def test_gsc0_equal_gengamma():
    gg = gengamma(a=0.75, c=2.0, scale=1.5)  # a = d / c, c = p
    gsc1 = gen_stable_count(alpha=0.0, sigma=1.5, d=1.5-2.0, p=2.0)
    gsc2 = gen_stable_count(alpha=0.1, sigma=1.5, d=1.5-2.0, p=2.0)
    x = gg.moment(1)
    _dual_compare(x, gg, gsc1, gsc2)


def test_gsc0_equal_invgamma():
    df = 4.0
    ig = invgamma(a=df)
    gsc1 = gen_stable_count(alpha=0.0, sigma=1.0,  d=-df+1, p=-1.0)  
    gsc2 = gen_stable_count(alpha=0.05, sigma=1.0,  d=-df+1, p=-1.0)  
    x = ig.moment(1)
    _dual_compare(x, ig, gsc1, gsc2)

    x = 0.85
    ig2 = invgamma(a=0.5, scale=0.25)
    ig2_gsc = stable_count.rv_stable_one_sided(0.5)
    compare_two_rvs(x, ig2, ig2_gsc, msg_prefix="rv_stable_one_sided")


def test_gsc0_equal_invweibull():
    df = 3.0
    iwb = invweibull(c=df)
    x = 0.85
    gsc1 = gen_stable_count(alpha=0.0,  sigma=1.0,  d=0.0, p=-df)  
    gsc2 = gen_stable_count(alpha=0.05, sigma=1.0,  d=0.0, p=-df)  
    _dual_compare(x, iwb, gsc1, gsc2)


# ----------------------------------------------------------------
# alpha = 1/2
def test_gsc_half_weibull():
    k = 0.56
    wb = weibull_min(c=k)
    x = wb.moment(1)
    gsc1 = gen_stable_count(alpha=0.5, sigma=2**(-2/k),  d=k/2, p=k/2)  
    compare_two_rvs(x, wb, gsc1, msg_prefix="weibull a=1/2")


def test_gsc_half_equal_gamma():
    a = 1.6
    sigma = 1.5
    gg = gengamma(a=a, c=1.0, scale=sigma)  # a = d
    x = gg.moment(1)
    gsc1 = gen_stable_count(alpha=0.5, sigma=sigma/4,  d=a-0.5, p=0.5)  
    compare_two_rvs(x, gg, gsc1, msg_prefix="gamma a=1/2")


def test_gsc_half_equal_chi():
    k = 3.0
    rv_chi = chi(df=k)
    x = rv_chi.moment(1)
    gsc1 = gen_stable_count(alpha=0.5, sigma=1/np.sqrt(2), d=k-1, p=1.0)  
    compare_two_rvs(x, rv_chi, gsc1, msg_prefix="chi a=1/2")


def test_gsc_half_equal_chi2():
    k = 2.8
    rv_chi2 = chi2(df=k)
    x = rv_chi2.moment(1)
    gsc1 = gen_stable_count(alpha=0.5, sigma=0.5, d=(k-1)/2, p=0.5)  
    compare_two_rvs(x, rv_chi2, gsc1, msg_prefix="chi2 a=1/2")


def test_gsc_half_equal_gengamma():
    s = 0.75
    c = 2.0
    sigma = 1.5
    gg = gengamma(a=s, c=c, scale=sigma)  # a = d / c, c = p
    x = gg.moment(1)
    gsc1 = gen_stable_count(alpha=0.5, sigma=2**(-2/c)*sigma, d=(s-0.5)*c, p=c/2)  
    compare_two_rvs(x, gg, gsc1, msg_prefix="gengamma a=1/2")


def test_gsc_half_equal_gengamma_v2():
    d = 3.1
    p = 2.0
    sigma = 1.5
    gg = gengamma(a=d/p, c=p, scale=sigma)  # a = d / c, c = p
    x = gg.moment(1)
    gsc1 = gen_stable_count(alpha=0.5, sigma=2**(-2/p)*sigma, d=d-p/2, p=p/2)  
    compare_two_rvs(x, gg, gsc1, msg_prefix="gengamma v2 a=1/2")


def test_gsc_half_equal_invgamma():
    df = 4.0
    ig = invgamma(a=df)
    x = ig.moment(1)
    gsc1 = gen_stable_count(alpha=0.5, sigma=4.0,  d=0.5-df, p=-0.5)  
    compare_two_rvs(x, ig, gsc1, msg_prefix="ig a=1/2")


def test_gsc_half_equal_invweibull():
    df = 3.0
    iwb = invweibull(c=df)
    x = 0.85
    gsc1 = gen_stable_count(alpha=0.5, sigma=2**(2/df),  d=-df/2, p=-df/2)  
    compare_two_rvs(x, iwb, gsc1, msg_prefix="iwb a=1/2")


# ----------------------------------------------------------------
# ----------------------------------------------------------------
# ----------------------------------------------------------------
# fractional
class TestFractionalMapping:
    alpha = 0.55
    x = 0.85

    la = stable_count.rv_stable_one_sided(alpha)
    la_gsc = gen_stable_count(alpha=alpha, sigma=1.0, d=0.0, p=-alpha)
    sc = stable_count(alpha)
    sc_gsc = gen_stable_count(alpha=alpha, sigma=1.0, d=1.0, p=alpha)
    F_gsc  = gen_stable_count(alpha=alpha, sigma=1.0, d=1.0, p=1.0)
    M_gsc  = gen_stable_count(alpha=alpha, sigma=1.0, d=0.0, p=1.0)

    def test_one_sided_stable_equal_gsc(self):
        alpha = self.alpha
        x = self.x
        p = compare_two_rvs(self.x, self.la, self.la_gsc)
        q = 1/x * wright_fn(-1/(x**alpha), -alpha, 0) 
        delta_precise_up_to(p, q)

    def test_sc_equal_gsc(self):
        compare_two_rvs(self.x, self.sc, self.sc_gsc, min_p=0.1)

    def test_sv_equal_gsc(self):
        alpha = 1.15  # this needs to be 2x larger
        sv = stable_vol(alpha)
        sv_gsc = gen_stable_count(alpha=alpha/2, sigma=1.0/np.sqrt(2.0), d=1.0, p=alpha)
        compare_two_rvs(self.x, sv, sv_gsc, min_p=0.1)
        
        x = self.x 
        wr = np.sqrt(2*np.pi)/gamma(1/alpha+1) * wright_fn(-(np.sqrt(2)*x)**alpha, -alpha/2, 0)
        delta_precise_up_to(wr, sv.pdf(x))

    def test_m_right_F_equal_gsc(self):
        alpha = self.alpha
        x = self.x
        p = wright_fn(-x, -alpha, 0) * gamma(alpha)
        q = self.F_gsc.pdf(x)
        assert p > 0.1
        delta_precise_up_to(p, q)

    def test_m_right_M_equal_gsc(self):
        alpha = self.alpha
        x = self.x
        p = wright_fn(-x, -alpha, 0) / x / alpha
        q = self.M_gsc.pdf(x)
        assert p > 0.1
        delta_precise_up_to(p, q)

    # below we are testing the wright functions, not distribution
    def test_m_right_f_equiv(self):
        alpha = self.alpha
        x = self.x
        p = wright_fn(-x, -alpha, 0)
        q = wright_f_fn(x, alpha)
        delta_precise_up_to(p, q)

    def test_m_right_f_equal_sc(self):
        alpha = self.alpha
        x = self.x
        p = wright_f_fn(x, alpha)
        q = wright_f_fn_by_sc(x, alpha)
        delta_precise_up_to(p, q)

    def test_m_right_m_equiv(self):
        alpha = self.alpha
        x = self.x
        p = wright_fn(-x, -alpha, 0) / x / alpha
        q = wright_m_fn(x, alpha)
        delta_precise_up_to(p, q)

    def test_m_right_m_alt_equiv(self):
        alpha = self.alpha
        x = self.x
        p = mainardi_wright_fn(x, alpha)
        q = wright_m_fn(x, alpha)
        delta_precise_up_to(p, q)

    # test W_{-a,-1}(-z)
    def test_mu_numerator_equiv(self):
        alpha = self.alpha
        x = self.x
        dx = x/1000
        p = wright_fn(-x, -alpha, -1.0)
        f = wright_f_fn_by_sc(x, alpha)
        f_dx =  wright_f_fn_by_sc(x+dx, alpha)
        q = -alpha * x * (f_dx - f)/dx - f
        
        delta_precise_up_to(p, q, abstol=0.01, reltol=0.001)
