
# test gsc and sc

import numpy as np
import pandas as pd

from scipy.special import gamma
from scipy.integrate import quad
from scipy.stats import gengamma

from .stable_count_dist import stable_count, gen_stable_count, stable_vol,\
    gsc_moment, gsc_mellin_transform,\
    gsc_normalization_constant, gsc_log_normalization_constant,\
    stable_count_pdf_small_x, stable_count_pdf_large_x, gsc_pdf_large_x,\
    gsc_mu_by_f, gsc_mu_by_m, gsc_mu_by_m_series, gsc_mu_at_half_alpha

from .wright import wright_fn, wright_m_fn_by_levy, wright_f_fn_by_levy
from .hankel import *
from .unit_test_utils import *
from .frac_gamma import frac_gamma_inc
from .mellin import pdf_by_mellin


alpha_0_55 = 0.55
sc_0_55 = sc = stable_count(alpha=alpha_0_55)


# ----------------------------------------------------------------
def test_sc_known_case():
    alpha = 0.5
    sc = stable_count(alpha)
    x = 0.85

    p = sc.pdf(x)  # type: ignore
    q = 1.0/(4 * np.sqrt(np.pi)) * x**0.5 * np.exp(-x/4)
    delta_precise_up_to(p, q)


def test_sc_asymp_exact():
    alpha = 0.5
    for x in [0.4, 0.88, 1.35]:
        p = stable_count(alpha).pdf(x)  # type: ignore
        q = stable_count_pdf_large_x(x, alpha)
        delta_precise_up_to(p, q)


def test_sc_asymp_large_x():
    alpha = 0.65
    for x in [10.0, 15.0, 20.0]:
        p = stable_count(alpha).pdf(x)  # type: ignore
        q = stable_count_pdf_large_x(x, alpha)
        delta_precise_up_to(p, q, abstol=0.005, reltol=0.005)


def test_sc_asymp_small_x():
    alpha = 0.45
    for x in [0.001, 0.002]:
        p = stable_count(alpha).pdf(x)  # type: ignore
        q = stable_count_pdf_small_x(x, alpha)
        delta_precise_up_to(p, q, abstol=0.005, reltol=0.02)


# ----------------------------------------------------------------
def test_sc_equal_gsc():
    alpha = 0.55
    x = 0.85
    sc = stable_count(alpha)
    gsc = gen_stable_count(alpha=alpha, sigma=1.0, d=1.0, p=alpha)
    compare_two_rvs(x, sc, gsc)
    
def test_sc_equal_gsc_cdf():
    alpha = 0.55
    x = 0.85
    sc = stable_count(alpha)
    gsc = gen_stable_count(alpha=alpha, sigma=1.0, d=1.0, p=alpha)
    p1 = sc.cdf(x)
    p2 = gsc.cdf(x)
    delta_precise_up_to(p1, p2)

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

    assert abs(np.imag(q1)) < 1e-8  # type: ignore
    delta_precise_up_to(p, q1)

    assert abs(np.imag(q2)) < 1e-8  # type: ignore
    delta_precise_up_to(p, q2)


# this is slow
def test_gsc_hankel():
    alpha = 0.75 
    sigma = 1.1
    d = 2.1
    p = 1.4
    
    x = 0.75
    gsc = gen_stable_count(alpha=alpha, sigma=sigma, d=d, p=p)
    p1 = gsc.pdf(x)  # type: ignore

    q = alpha * d / p
    g = gamma(q)
    def gsc_integrand(t):
        e_term = g * np.exp(t) / t**q
        scale = sigma/t**(alpha/p)
        return e_term * pdf_gg(x, a=scale, d=d, p=p)

    q1 = hankel_integral_mpr(gsc_integrand)  # parallel version

    assert abs(np.imag(q1)) < 1e-8  # type: ignore
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
    
    c = gsc_normalization_constant(alpha, sigma, d, p)
    log_c = gsc_log_normalization_constant(alpha, np.log(sigma), d, p)

    c_d0 = gsc_normalization_constant(alpha, sigma, 0.0, p)
    log_c_d0 = gsc_log_normalization_constant(alpha, np.log(sigma), 0.0, p)

    def test_gsc_constant(self):
        p1 = self.c
        p2 = np.exp(self.log_c)
        delta_precise_up_to(p1, p2)

    def test_gsc_constant_d0(self):
        p1 = self.c_d0
        p2 = np.exp(self.log_c_d0)
        delta_precise_up_to(p1, p2)

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
        
        def fn1(x): return x * self.gsc_d0.pdf(x)  # type: ignore
        
        m2 = quad(fn1, a=0.001, b=np.inf, limit=10000)[0]
        delta_precise_up_to(m1, m2)
        
    def test_mnt2_d0(self):
        m1 = gsc_moment(2.0, alpha=self.alpha, sigma=self.sigma, d=0.0, p=self.p)
        
        def fn2(x): return x**2 * self.gsc_d0.pdf(x)  # type: ignore
        
        m2 = quad(fn2, a=0.001, b=np.inf, limit=10000)[0]
        delta_precise_up_to(m1, m2)

    def test_mnt1_a1(self):
        m1 = gsc_moment(1.0, alpha=1.0, sigma=self.sigma, d=self.d, p=self.p)
        
        def fn1(x): return x * self.gsc_a1.pdf(x)  # type: ignore
        
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
            p1 = self.gsc.pdf(x)  # type: ignore
            p2 = gsc_pdf_large_x(x, alpha=self.alpha, sigma=self.sigma, d=self.d, p=self.p)
            assert p1 > 1e-8
            delta_precise_up_to(p1, p2, abstol=0.005, reltol=0.005)


# ----------------------------------------------------------------
# gsc moments
class Test_GSC_Mellin:
    alpha = 0.75 
    sigma = 1.1
    d = 2.1
    p = 1.4

    x = 0.45
    
    gsc = gen_stable_count(alpha=alpha, sigma=sigma, d=d, p=p)
    gsc_a0 = gen_stable_count(alpha=0.0, sigma=sigma, d=d, p=p)
    gg0 = gg_rv(a=sigma, d=d+p, p=p)

    gsc_d0 = gen_stable_count(alpha=alpha, sigma=sigma, d=0.0, p=p)
    
    def equal_wright_f_fn(self):
        p1 = wright_f_fn_by_levy(self.x, self.alpha)
        p2 = pdf_by_mellin(
            self.x, lambda s: gsc_mellin_transform(s, self.alpha, 1.0, 1.0, 1.0)) / gamma(self.alpha)
        delta_precise_up_to(p1, p2)

    def equal_wright_m_fn(self):
        p1 = wright_m_fn_by_levy(self.x, self.alpha)
        p2 = pdf_by_mellin(
            self.x, lambda s: gsc_mellin_transform(s, self.alpha, 1.0, 0.0, 1.0))
        delta_precise_up_to(p1, p2)

    def test_gsc(self):
        p1 = pdf_by_mellin(
            self.x, lambda s: gsc_mellin_transform(s, self.alpha, self.sigma, self.d, self.p))
        p2 = self.gsc.pdf(self.x)  # type: ignore
        delta_precise_up_to(p1, p2)

    def test_gsc_a0(self):
        p1 = pdf_by_mellin(
            self.x, lambda s: gsc_mellin_transform(s, 0.0, self.sigma, self.d, self.p))
        p2 = self.gsc_a0.pdf(self.x)  # type: ignore
        p3 = self.gg0.pdf(self.x)  # type: ignore
        delta_precise_up_to(p1, p2)
        delta_precise_up_to(p1, p3)
    
    def test_gsc_d0(self):
        p1 = pdf_by_mellin(
            self.x, lambda s: gsc_mellin_transform(s, self.alpha, self.sigma, 0.0, self.p))
        p2 = self.gsc_d0.pdf(self.x)  # type: ignore
        delta_precise_up_to(p1, p2)


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


def test_gsc_cdf():
    alpha = 0.6 
    sigma = 0.7

    p = 0.45  
    d = 0.55 

    x = 0.95 

    g = gen_stable_count(alpha, sigma, d, p)
    p1 = g.cdf(x)

    # the book shies away from using frac_gamma_inc, it just uses frac_gamma_star directly
    # but we can keep the test here
    s2 = d/(2*p) + 0.5 
    sigma2 = sigma * 2**(1/p)
    x2 = (x/sigma2)**(2*p) 
    p2 = frac_gamma_inc(s2, x2, alpha)
    delta_precise_up_to(p1, p2)

    def _cdf(x):
        return quad(lambda s: g.pdf(s), 0, x, limit=200)[0]  # type: ignore
    p3 = _cdf(x)
    delta_precise_up_to(p1, p3)
