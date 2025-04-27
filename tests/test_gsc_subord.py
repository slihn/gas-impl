
# test gsc and sc
# pyright: reportGeneralTypeIssues=false

import numpy as np
import pandas as pd

from scipy.special import gamma, erfc
from scipy.integrate import quad
from scipy import stats
from scipy.stats import gengamma, norm, expon, weibull_min, rayleigh, poisson, chi, chi2, invgamma, invweibull

from .stable_count_dist import stable_count, gen_stable_count, gsc_normalization_constant, stable_vol
from .wright import wright_fn, mittag_leffler_fn, wright_m_fn
from .hankel import *
from .unit_test_utils import *


def weibull_gsc(df, precise=True):
    alpha = 0.0 if precise else 0.1
    return gen_stable_count(alpha=alpha, sigma=1.0, d=0.0, p=df)

def _norm(x): return norm().pdf(x) 


# ----------------------------------------------------------------
# ----------------------------------------------------------------
# ----------------------------------------------------------------
# product


class TestStretched:
    alpha = 0.55
    x = 0.2
    p1 = stretched1(x, alpha)
    assert p1 > 0.2
    stretched1_gsc   = gen_stable_count(alpha=0.0,     sigma=1.0,            d=1.0-alpha, p=alpha)
    sc_gsc           = gen_stable_count(alpha=alpha,   sigma=1.0,            d=1.0, p=alpha)
    sv_gsc           = gen_stable_count(alpha=alpha/2, sigma=1.0/np.sqrt(2), d=1.0, p=alpha)

    def perform_product_test(self, unit_fn, gsc):
        product_dist_test_suite(self.p1, self.x, unit_fn, gsc)

    def test_stretched_vs_sc(self):
        p2 = self.stretched1_gsc.pdf(self.x)
        delta_precise_up_to(self.p1, p2)
        self.perform_product_test(laplace1, self.sc_gsc)

    def test_stretched_vs_sv(self):
        def half_norm(x): return 2.0*norm.pdf(x)
        self.perform_product_test(half_norm, self.sv_gsc)

    def test_stretched_k_vs_gsc(self):
        k = 3
        def stretched1_k(x): return stretched1(x, k)
        gsc = gen_stable_count(alpha=self.alpha/k, sigma=1.0, d=1.0, p=self.alpha)
        self.perform_product_test(stretched1_k, gsc)


class TestWeibull:
    k = 0.55  # this is k in Wb(x;k)
    x = 0.2
    p1 = weibull_min(c=k).pdf(x)
    assert p1 > 0.2
    weibull_gsc = weibull_gsc(k)
    sc_v2_gsc   = gen_stable_count(alpha=k,   sigma=1.0,            d=0.0, p=k)
    sv_v2_gsc   = gen_stable_count(alpha=k/2, sigma=1.0/np.sqrt(2), d=0.0, p=k)

    def perform_product_test(self, unit_fn, gsc):
        product_dist_test_suite(self.p1, self.x, unit_fn, gsc)

    def test_weibull_vs_sc(self):
        p2 = self.weibull_gsc.pdf(self.x)
        delta_precise_up_to(self.p1, p2)
        self.perform_product_test(laplace1, self.sc_v2_gsc)

    def test_weibull_vs_sv(self):
        def _rayleigh(x): return rayleigh.pdf(x)
        self.perform_product_test(_rayleigh, self.sv_v2_gsc)


def test_gamma_gsc_product():
    s = 4.0  # this is s in Gamma(x;s)
    x = 1.8

    gamma_gsc        = gen_stable_count(alpha=0.0,    sigma=1.0, d=s-1, p=1.0)
    stable_gamma_gsc = gen_stable_count(alpha=1.0/s,  sigma=1.0, d=s,   p=1.0)  # sigma, d, p are preserved

    gg = stats.gamma(a=s)
    p1 = compare_two_rvs(x, gg, gamma_gsc, min_p=0.1)

    def _weibull(x): return weibull_min(c=s).pdf(x)
    product_dist_test_suite(p1, x, _weibull, stable_gamma_gsc)


def test_poisson_gsc_product():
    k  = 4.0  # occurence
    mu = 3.0  # lambda

    poisson_gsc        = gen_stable_count(alpha=0.0,       sigma=1.0, d=k,   p=1.0)
    stable_poisson_gsc = gen_stable_count(alpha=1.0/(k+1), sigma=1.0, d=k+1, p=1.0)  # sigma, d, p are preserved

    p1 = poisson(mu=mu).pmf(k)
    p2 = poisson_gsc.pdf(mu)
    assert p1 > 0.1
    delta_precise_up_to(p1, p2)

    def _weibull(x): return weibull_min(c=k+1).pdf(x)
    product_dist_test_suite(p1, mu, _weibull, stable_poisson_gsc)


class TestChi_Chi2:
    k = 4.0  # k
    x = 0.8
    p1 = chi(df=k).pdf(x)
    p2 = chi2(df=k).pdf(x)
    assert p1 > 0.1
    assert p2 > 0.1
    chi_gsc         = gen_stable_count(alpha=0.0,   sigma=np.sqrt(2),  d=k-2,   p=2.0)
    chi2_gsc        = gen_stable_count(alpha=0.0,   sigma=2.0,         d=k/2-1, p=1.0)
    stable_chi_gsc  = gen_stable_count(alpha=2.0/k, sigma=np.sqrt(2),  d=k,     p=2.0)  # sigma, d, p are preserved
    stable_chi2_gsc = gen_stable_count(alpha=2.0/k, sigma=2.0,         d=k/2,   p=1.0)  # sigma, d, p are preserved

    def test_chi_gsc(self):
        p3 = self.chi_gsc.pdf(self.x)
        delta_precise_up_to(self.p1, p3)

        def _weibull(x): return weibull_min(c=self.k).pdf(x)
        product_dist_test_suite(self.p1, self.x, _weibull, self.stable_chi_gsc)

    def test_chi2_gsc(self):
        p3 = self.chi2_gsc.pdf(self.x)
        delta_precise_up_to(self.p2, p3)

        def _weibull(x): return weibull_min(c=self.k/2).pdf(x)
        product_dist_test_suite(self.p2, self.x, _weibull, self.stable_chi2_gsc)


class TestGenGamma:
    df = 4.0
    p = 2.5
    x = 1.8
    p1 = stats.gengamma(a=df/p, c=p).pdf(x)
    assert p1 > 0.2
    gengamma_gsc        = gen_stable_count(alpha=0.0,  sigma=1.0, d=df-p, p=p)
    stable_gengamma_gsc = gen_stable_count(alpha=p/df, sigma=1.0, d=df,   p=p)  # sigma, d, p are preserved

    def perform_product_test(self, unit_fn, gsc):
        product_dist_test_suite(self.p1, self.x, unit_fn, gsc)

    def test_gengamma_gsc(self):
        p2 = self.gengamma_gsc.pdf(self.x)
        delta_precise_up_to(self.p1, p2)

        def _weibull(x): return weibull_min(c=self.df).pdf(x)
        self.perform_product_test(_weibull, self.stable_gengamma_gsc)


# ----------------------------------------------------------------
# ----------------------------------------------------------------
# ----------------------------------------------------------------
# ratio


def test_invgamma_gsc_ratio():
    k = 4.0
    x = 0.25

    ig = invgamma(k)
    ig_gsc        = gen_stable_count(alpha=0.0, sigma=1.0, d=-k+1, p=-1.0)  
    stable_ig_gsc = gen_stable_count(alpha=1/k, sigma=1.0, d=k,    p=1.0)  
    p1 = compare_two_rvs(x, ig, ig_gsc)

    def _iwb(x): return invweibull(k).pdf(x) 
    ratio_dist_test_suite(p1, x, _iwb, stable_ig_gsc)


def test_invweibull_gsc_ratio():
    alpha = 0.85
    x = 0.25
    iwb = invweibull(alpha)
    p1 = iwb.pdf(x)

    def _iwb1(x): return invweibull(1).pdf(x) 
    def _iwb2(x): return invweibull(2).pdf(x) 

    stable_iwb1_gsc    = gen_stable_count(alpha=alpha,   sigma=1.0,  d=0.0,  p=alpha)  
    stable_iwb2_gsc    = gen_stable_count(alpha=alpha/2, sigma=1.0,  d=0.0,  p=alpha)  
    ratio_dist_test_suite(p1, x, _iwb1, stable_iwb1_gsc)
    ratio_dist_test_suite(p1, x, _iwb2, stable_iwb2_gsc)


def test_student_t_gsc_ratio():
    k = 3.0
    x = 0.5
    t = stats.t(k)
    p1 = t.pdf(x)
 
    stable_t_gsc    = gen_stable_count(alpha=0.5,   sigma=1/np.sqrt(2*k),  d=k-1,  p=1.0)  
    ratio_dist_test_suite(p1, x, _norm, stable_t_gsc)


def test_mittag_leffler_half():
    alpha = 0.5
    x = 0.45
    p1 = mittag_leffler_fn(x, alpha=alpha)
    p2 = np.exp(x**2) * erfc(-x)
    delta_precise_up_to(p1, p2)


class Test_Mittag_Leffler: 
    # test the laplace transform of M-Wright to MLF
    alpha = 0.55
    x = 0.25

    m_gsc_d0     = gen_stable_count(alpha=alpha,   sigma=1.0, d=0.0, p=1.0) 
    m_gsc_d1     = gen_stable_count(alpha=alpha,   sigma=1.0, d=-0.9999, p=1.0)   # d -> -1 but can not be at -1
    C1 = gsc_normalization_constant(alpha=alpha,   sigma=1.0, d=-0.9999, p=1.0)   # C = 0 at d = -1, watch out

    def test_gsc_m_wright_equiv(self):
        q1 = wright_m_fn(self.x, alpha=self.alpha)
        q2 = self.m_gsc_d0.pdf(self.x)
        delta_precise_up_to(q1, q2)
        
    def test_gsc_m_wright_equiv_v2(self):
        q1 = wright_m_fn(self.x, alpha=self.alpha)
        q2 = self.m_gsc_d1.pdf(self.x) * self.x / ( self.C1 * self.alpha )
        delta_precise_up_to(q1, q2, abstol=0.001, reltol=0.0005)

    def test_gsc_integral_v1(self):
        p1 = mittag_leffler_fn(-self.x, alpha=self.alpha)
        
        def fn1(s):
            # ML's Laplace transform
            return np.exp(-self.x*s) *  self.m_gsc_d0.pdf(s)

        p2 = quad(fn1, a=0, b=np.inf, limit=1000)[0]
        delta_precise_up_to(p1, p2)

    def test_gsc_integral_v2(self):
        p1 = mittag_leffler_fn(-self.x, alpha=self.alpha)

        def fn2(s):
            # ML's Laplace transform forced into a ratio distribution form
            return s * np.exp(-self.x*s) * self.m_gsc_d1.pdf(s) 

        p2 = quad(fn2, a=0.0, b=np.inf, limit=100000)[0] / ( self.C1 * self.alpha )
        delta_precise_up_to(p1, p2)

    def test_gsc_integral_v3(self):
        # The second normal-like ML formula. This doesn't need any trick
        alpha = self.alpha
        q = gamma(1.0-alpha/4) / (2.0**0.5 * np.pi) 
        p1 = q * mittag_leffler_fn(-self.x**2/2, alpha=alpha/2) 

        m2_gsc = gen_stable_count(alpha=alpha/2, sigma=1.0, d=-1.0, p=2.0) 
        ratio_dist_test_suite(p1, self.x, _norm, m2_gsc)


# SaS and GSaS ratio tests are in GSaS file
