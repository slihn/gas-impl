
# test gsc and sc

import numpy as np
import pandas as pd
from numba import njit  # this causes an error: DeprecationWarning: `np.MachAr` is deprecated (NumPy 1.22).

from scipy.special import gamma
from scipy.integrate import quad
from scipy.stats import gengamma

from gas_impl.stable_count_dist import stable_count, gen_stable_count, stable_vol, gsc_moment,\
    stable_count_pdf_small_x, stable_count_pdf_large_x, gsc_pdf_large_x,\
    gsc_mu_by_f, gsc_mu_by_m, gsc_mu_by_m_series, gsc_mu_at_half_alpha

from gas_impl.wright import wright_fn
from gas_impl.hankel import *
from gas_impl.unit_test_utils import *


alpha_0_55 = 0.55
sc_0_55 = sc = stable_count(alpha=alpha_0_55)


# ----------------------------------------------------------------
def test_sc_known_case():
    alpha = 0.5
    sc = stable_count(alpha)
    x = 0.85

    p = sc.pdf(x)
    q = 1.0/(4 * np.sqrt(np.pi)) * x**0.5 * np.exp(-x/4)
    delta_precise_up_to(p, q)


def test_sc_asymp_exact():
    alpha = 0.5
    for x in [0.4, 0.88, 1.35]:
        p = stable_count(alpha).pdf(x)
        q = stable_count_pdf_large_x(x, alpha)
        delta_precise_up_to(p, q)


def test_sc_asymp_large_x():
    alpha = 0.65
    for x in [10.0, 15.0, 20.0]:
        p = stable_count(alpha).pdf(x)
        q = stable_count_pdf_large_x(x, alpha)
        delta_precise_up_to(p, q, abstol=0.005, reltol=0.005)


def test_sc_asymp_small_x():
    alpha = 0.45
    for x in [0.001, 0.002]:
        p = stable_count(alpha).pdf(x)
        q = stable_count_pdf_small_x(x, alpha)
        delta_precise_up_to(p, q, abstol=0.005, reltol=0.02)


# ----------------------------------------------------------------
def test_sc_equal_gsc():
    alpha = 0.55
    x = 0.85
    sc = stable_count(alpha)
    gsc = gen_stable_count(alpha=alpha, sigma=1.0, d=1.0, p=alpha)
    compare_two_rvs(x, sc, gsc)

def test_moments_sc_equal_gsc():
    for alpha in [0.45, 0.5, 0.65]:
        sc = stable_count(alpha)
        gsc = gen_stable_count(alpha=alpha, sigma=1.0, d=1.0, p=alpha)
        for n in [1,2,3,4]:
            m1 = sc.moment(float(n))
            m2 = gsc.moment(float(n))
            delta_precise_up_to(m1, m2)

def test_sv_equal_gsc():
    alpha = 0.85
    x = 1.15
    sv = stable_vol(alpha)
    gsc = gen_stable_count(alpha=alpha/2, sigma=1.0/np.sqrt(2), d=1.0, p=alpha)
    compare_two_rvs(x, sv, gsc)

def test_gsc0_equal_gengamma():
    x = 0.85
    s = 3.5
    c = 2.1
    gg = gengamma(a=s, c=c)
    gsc = gen_stable_count(alpha=0.0, sigma=1.0, d=c*(s-1), p=c)
    compare_two_rvs(x, gg, gsc)


# ----------------------------------------------------------------
# test the hankel utilities
def test_m_wright_hankel():
    alpha = 0.55 
    x = 0.85
    p = wright_fn(-x, -alpha, 0.0)  # F_alpha(x)

    def wright_integrand(t):
        return np.exp(t - x * t**alpha)

    q1 = hankel_integral(wright_integrand)
    q2 = hankel_integral_mpr(wright_integrand)  # parallel version

    assert abs(np.imag(q1)) < 1e-8
    delta_precise_up_to(p, q1)

    assert abs(np.imag(q2)) < 1e-8
    delta_precise_up_to(p, q2)


# this is slow
def test_gsc_hankel():
    alpha = 0.75 
    sigma = 1.1
    d = 2.1
    p = 1.4
    
    x = 0.75
    gsc = gen_stable_count(alpha=alpha, sigma=sigma, d=d, p=p)
    p1 = gsc.pdf(x)

    q = alpha * d / p
    g = gamma(q)
    def gsc_integrand(t):
        e_term = g * np.exp(t) / t**q
        scale = sigma/t**(alpha/p)
        return e_term * pdf_gg(x, a=scale, d=d, p=p)

    q1 = hankel_integral_mpr(gsc_integrand)  # parallel version

    assert abs(np.imag(q1)) < 1e-8
    delta_precise_up_to(p1, q1)

# ----------------------------------------------------------------
# gsc moments
class Test_GSC_Moments:
    alpha = 0.75 
    sigma = 1.1
    d = 2.1
    p = 1.4
    
    gsc = gen_stable_count(alpha=alpha, sigma=sigma, d=d, p=p)
    gsc0 = gen_stable_count(alpha=0.0, sigma=sigma, d=d, p=p)
    gg0 = gg_rv(a=sigma, d=d+p, p=p)

    gsc_d0 = gen_stable_count(alpha=alpha, sigma=sigma, d=0.0, p=p)
    gsc_a1 = gen_stable_count(alpha=0.99,  sigma=sigma, d=d,   p=p)  # delta function

    def mnt(self, n):
        return gsc_moment(n, alpha=self.alpha, sigma=self.sigma, d=self.d, p=self.p)
    
    def test_moment_fn(self):
        for n in [1,2,3,4]:
            p1 = self.mnt(float(n))
            p2 = self.gsc.moment(float(n))
            delta_precise_up_to(p1, p2)

    def test_moment_fn_gg(self):
        for n in [1,2,3,4]:
            p1 = self.gg0.moment(float(n))
            p2 = self.gsc0.moment(float(n))
            delta_precise_up_to(p1, p2, msg_prefix=f"{n}-th moment")
            
    def test_mnt1_d0(self):
        m1 = gsc_moment(1.0, alpha=self.alpha, sigma=self.sigma, d=0.0, p=self.p)
        
        def fn1(x): return x * self.gsc_d0.pdf(x)
        
        m2 = quad(fn1, a=0.001, b=np.inf, limit=10000)[0]
        delta_precise_up_to(m1, m2)
        
    def test_mnt2_d0(self):
        m1 = gsc_moment(2.0, alpha=self.alpha, sigma=self.sigma, d=0.0, p=self.p)
        
        def fn2(x): return x**2 * self.gsc_d0.pdf(x)
        
        m2 = quad(fn2, a=0.001, b=np.inf, limit=10000)[0]
        delta_precise_up_to(m1, m2)

    def test_mnt1_a1(self):
        m1 = gsc_moment(1.0, alpha=1.0, sigma=self.sigma, d=self.d, p=self.p)
        
        def fn1(x): return x * self.gsc_a1.pdf(x)
        
        m2 = quad(fn1, a=0.001, b=np.inf, limit=10000)[0]
        delta_precise_up_to(m1, self.sigma)
        delta_precise_up_to(m1, m2, abstol=0.02, reltol=0.02)

    def test_a0_d0(self):
        for n in [1,2,3,4]:
            p1 = gsc_moment(float(n), alpha=0.0, sigma=self.sigma, d=0.0, p=self.p)
            p2 = gg_rv(a=self.sigma, d=self.p, p=self.p).moment(float(n))
            delta_precise_up_to(p1, p2, msg_prefix=f"{n}-th moment")

    def test_gsc_pdf_large_x(self):
        for x in [2.5, 2.6]:
            p1 = self.gsc.pdf(x)
            p2 = gsc_pdf_large_x(x, alpha=self.alpha, sigma=self.sigma, d=self.d, p=self.p)
            assert p1 > 1e-8
            delta_precise_up_to(p1, p2, abstol=0.005, reltol=0.005)


# ----------------------------------------------------------------
# gsc mu
class Test_GSC_Mu:
    alpha = 0.46 
    sigma = 1.1
    d = 1.3
    p = alpha*2
    
    x = 0.5
    mu1 = gsc_mu_by_f(x, dz_ratio=None, alpha=alpha, sigma=sigma, d=d, p=alpha)
    mu2 = gsc_mu_by_f(x, dz_ratio=0.0001, alpha=alpha, sigma=sigma, d=d, p=alpha)
    mu3 = gsc_mu_by_m(x, dz_ratio=0.0001, alpha=alpha, sigma=sigma, d=d, p=alpha)
    mu4 = gsc_mu_by_m_series(x, alpha=alpha, sigma=sigma, d=d, p=alpha)

    def test_mu_by_f(self):
        delta_precise_up_to(self.mu1, self.mu2)

    def test_mu_three_ways(self):
        delta_precise_up_to(self.mu2, self.mu3)
        delta_precise_up_to(self.mu2, self.mu4)

    def test_sc_one_half(self):
        x = 0.55
        alpha = 0.5
        mu1 = gsc_mu_by_m_series(x, alpha=alpha, sigma=1.0, d=1.0, p=alpha)
        mu2 = (6-x)/8
        delta_precise_up_to(mu1, mu2)

        mu3 = gsc_mu_at_half_alpha(x, sigma=1.0, d=1.0, p=alpha)
        delta_precise_up_to(mu1, mu3)
        
    def test_sv_one_half(self):
        x = 0.55
        alpha = 0.5
        mu1 = gsc_mu_by_m_series(x, alpha=alpha, sigma=2**-0.5, d=1.0, p=alpha*2)
        mu2 = 1 - x**2/2
        delta_precise_up_to(mu1, mu2)

        mu3 = gsc_mu_at_half_alpha(x, sigma=2**-0.5, d=1.0, p=alpha*2)
        delta_precise_up_to(mu1, mu3)
