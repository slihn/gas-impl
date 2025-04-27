
# test gsas
# pyright: reportGeneralTypeIssues=false

import numpy as np
import pandas as pd
import mpmath as mp
from scipy.special import gamma, hyp2f1, beta
from scipy.integrate import quad
from scipy import stats
from scipy.stats import norm, cauchy, levy_stable

from .stable_count_dist import gen_stable_count
from .gas_dist import gsas, gsas_pdf_at_zero, gsas_std_pdf_at_zero, gsas_moment, gsas_kurtosis,\
    frac_hyp_fn, t_cdf_by_hyp2f1, t_cdf_by_betainc, t_cdf_by_binom
from .fcm_dist import fcm_moment, fcm_sigma
from .gsas_dist import GSaS_Wright, GSaS_Wright_MP, Wright4Ways_MP
from .hyp_geo2 import frac_hyp2f1_by_alpha_k
from .gexppow_dist import exppow
from .unit_test_utils import *


# -------------------------------------------------------------------------------------
def _cauchy(x): return cauchy().pdf(x) 
def _norm(x): return norm().pdf(x) 


# -------------------------------------------------------------------------------------
# basic stuff
class Test_T_Equiv:
    # P_1,k = t_k
    k = 3.0
    x = 0.85
    t_gsas = gsas(alpha=1.0, k=k)

    def test_pdf(self):
        compare_two_rvs(self.x, stats.t(self.k), self.t_gsas)

    def test_cdf(self):
        compare_cdf_of_two_rvs(self.x, stats.t(self.k), self.t_gsas)


# -------------------------------------------------------------------------------------
class Test_Frac_HyperGeo:
    alpha = 1.2
    k = 2.6
    g = gsas(alpha, k)
    fh = frac_hyp2f1_by_alpha_k(alpha, k, 0.5, 1.5)
    
    def test_gsas_cdf(self):
        for x in [-0.25, 0.0, 0.65, 1.5]:
            p1 = self.g.cdf(x)
            x1 = x / np.sqrt(self.k)
            p2 = 0.5 + x1 * frac_hyp_fn(-x1**2, self.alpha, self.k, 0.5, 1.5) 
            delta_precise_up_to(p1, p2, msg_prefix=f"x={x}")

    def test_hyp2f1_equiv(self):
        k = self.k
        b = 0.6
        c = 1.45

        b2 = beta(k/2, 1.0/2)
        for x in [-0.25, -0.45]:
            p1 = hyp2f1(b, (k+1)/2, c, x) 
            p2 = frac_hyp_fn(x, 1.0, k, b, c) * b2
            delta_precise_up_to(p1, p2, msg_prefix=f"x={x}")

    def test_gsas_cdf_by_frac_hyp2f1(self):
        for x in [-0.25, 0.0, 0.65, 1.5]:
            p1 = self.g.cdf(x)
            p2 = 0.5 + x / np.sqrt(2*np.pi) * self.fh.scaled_integral(-0.5 * x**2)
            delta_precise_up_to(p1, p2, msg_prefix=f"x={x}")


# Student's t CDF's relation to hyp2f1, betainc
class Test_T_CDF_Special_Fn:
    k = 2.5
    g = gsas(alpha=1.0, k=k)  # t
    
    def test_t_cdf_by_hyp2f1(self):
        for x in [-0.25, 0.0, 0.65, 1.5]:
            assert x**2 < self.k
            p1 = t_cdf_by_hyp2f1(x, self.k)
            p2 = stats.t(self.k).cdf(x)
            delta_precise_up_to(p1, p2, msg_prefix=f"x={x}")

            p3 = self.g.cdf(x)
            delta_precise_up_to(p2, p3, msg_prefix=f"x={x}")

    def test_t_cdf_by_betainc(self):
        for x in [-0.28, 0.62, 0.0, 2.0]:
            p1 = t_cdf_by_betainc(x, self.k)
            p2 = stats.t(self.k).cdf(x)
            delta_precise_up_to(p1, p2, msg_prefix=f"x={x}")

            p3 = self.g.cdf(x)
            delta_precise_up_to(p2, p3, msg_prefix=f"x={x}")

    def test_t_cdf_by_betainc_variant(self):
        for x in [-0.28, 0.62, 0.0, 2.0]:
            p1 = t_cdf_by_betainc(x, self.k, use_variant=True)
            p2 = stats.t(self.k).cdf(x)
            delta_precise_up_to(p1, p2, msg_prefix=f"x={x}")

            p3 = self.g.cdf(x)
            delta_precise_up_to(p2, p3, msg_prefix=f"x={x}")

    def test_t_cdf_by_binom(self):
        k = 5.0
        for x in [-0.28, 0.62, 0.0, 2.0]:
            p1 = t_cdf_by_binom(x, k)
            p2 = stats.t(k).cdf(x)
            delta_precise_up_to(p1, p2, msg_prefix=f"x={x}")


class Test_SaS_Equiv():
    alpha = 0.55
    x = 0.85
    sas_levy = levy_stable(alpha=alpha, beta=0)
    sas_gsas = gsas(alpha=alpha, k=1.0)

    def test_pdf(self):
        compare_two_rvs(self.x, self.sas_levy, self.sas_gsas, min_p=0.1)

    def test_cdf(self):
        compare_cdf_of_two_rvs(self.x, self.sas_levy, self.sas_gsas)


# -------------------------------------------------------------------------------------
# SaS ratio tests
class TestSaS_Ratio:
    alpha = 0.55
    x = 0.5
    sym_stable  = levy_stable(alpha=alpha, beta=0, scale=1.0)
    p1 = sym_stable.pdf(x)
    
    stable_sas_gsc_v1    = gen_stable_count(alpha=alpha,   sigma=1.0,             d=0.0,  p=alpha)  
    stable_sas_gsv_v2    = gen_stable_count(alpha=alpha/2, sigma=1.0/np.sqrt(2),  d=0.0,  p=alpha)

    def perform_ratio_test(self, unit_fn, gsc):
        ratio_dist_test_suite(self.p1, self.x, unit_fn, gsc)

    def test_sas_gsc_ratio_v1(self):
        self.perform_ratio_test(_cauchy, self.stable_sas_gsc_v1)

    def test_sas_gsc_ratio_v2(self):
        self.perform_ratio_test(_norm, self.stable_sas_gsv_v2)


# GSaS
def test_sas_pdf_at_zero_equiv():
    alpha = 0.75
    x = 0.0
    sas_levy = levy_stable(alpha=alpha, beta=0)
    g = gsas(alpha=alpha, k=1.0)
    compare_two_rvs(x, sas_levy, g, min_p=0.1)

    p1 = gsas_pdf_at_zero(alpha=alpha, k=1.0)
    p2 = gamma(1/alpha+1) / np.pi
    delta_precise_up_to(p1, p2)
    
    delta_precise_up_to(p1, sas_levy.pdf(0))


def test_exppow_pdf_at_zero_equiv():
    alpha = 0.75
    x = 0.0
    ep = exppow(alpha=alpha)
    g = gsas(alpha=alpha, k=-1.0)
    compare_two_rvs(x, ep, g, min_p=0.1)

    p1 = gsas_pdf_at_zero(alpha=alpha, k=-1.0)
    p2 = 0.5 / gamma(1/alpha+1)
    delta_precise_up_to(p1, p2)
    
    delta_precise_up_to(p1, ep.pdf(0))


class Test_GSAS_PDF0:
    alpha = 0.75
    k = 3.1
    g = gsas(alpha=alpha, k=k)

    def test_pdf_at_zero(self):
        p1 = gsas_pdf_at_zero(alpha=self.alpha, k=self.k)
        p2 = self.g.pdf(0.0)
        delta_precise_up_to(p1, p2)

    def test_std_pdf_at_zero(self):
        p1 = gsas_std_pdf_at_zero(alpha=self.alpha, k=self.k)
        sd = gsas_moment(n=2.0, alpha=self.alpha, k=self.k)**0.5
        p2 = gsas(alpha=self.alpha, k=self.k, scale=1/sd).pdf(0.0)
        delta_precise_up_to(p1, p2)

    def test_cdf_interval(self):
        x = 0.2
        def _kernel(z): return self.g.pdf(z)
        p1 = quad(_kernel, a=-x, b=x, limit=10000)[0]
        p2 = self.g.cdf(x) - self.g.cdf(-x)
        delta_precise_up_to(p1, p2)


class Test_GSAS_Reflection:
    alpha = 0.75
    k = 3.1

    def _moment_reflection(self, n):
        alpha = self.alpha
        k = self.k

        c = 2**(n/2) / np.sqrt(np.pi) * gamma((n+1)/2)
        p1 = gsas_moment(n, alpha=alpha, k=-k)
        p2 = c * fcm_moment(-n, alpha=alpha, k=-k)
        delta_precise_up_to(p1, p2, msg_prefix=f"n={n} form1")

        c = fcm_moment(n+1, alpha=alpha, k=k) / fcm_moment(-n, alpha=alpha, k=k) / fcm_moment(1.0, alpha=alpha, k=k)
        p2 = c * gsas_moment(n, alpha=alpha, k=k)
        delta_precise_up_to(p1, p2, msg_prefix=f"n={n} form2")

        c = fcm_moment(n+1, alpha=alpha, k=k) / fcm_moment(n+1, alpha=alpha, k=-k) / fcm_moment(1.0, alpha=alpha, k=k)**2
        p2 = c * gsas_moment(n, alpha=alpha, k=k)
        delta_precise_up_to(p1, p2, msg_prefix=f"n={n} form3")

    def test_moment_reflection_n2(self): self._moment_reflection(n = 2.0)
    def test_moment_reflection_n4(self): self._moment_reflection(n = 4.0)


# moments are tested in test_hard_gas.py
def test_gsas_kurtosis():
    for alpha in [0.75, 1.0, 1.25]:
        for k in [3.1, 4.3, 5.2]:
            p1 = gsas_kurtosis(alpha, k, exact_form=True)
            p2 = gsas_kurtosis(alpha, k, exact_form=False)
            delta_precise_up_to(p1, p2)



def _fcm_m2(alpha, k):
    alpha = mp.mpf(alpha)
    k = mp.mpf(k)
    m2 = alpha**(-2/alpha * mp.sign(k))
    return float(m2)

# GSaS CLM lemma
def _gsas_m2(alpha, k):
    alpha = mp.mpf(alpha)
    k = mp.mpf(k)
    m2 = alpha**(2/alpha * mp.sign(k))
    return float(m2)

class Test_GSaS_Var_LargeK:
    def test_gsas_variance(self):
        alpha = 0.85
        k = 3.5
        p1 = gsas_moment(2.0, alpha=alpha, k=k)
        p2 = fcm_moment(-2.0, alpha=alpha, k=k)
        delta_precise_up_to(p1, p2)
        
    def _var(self, alpha):
        with mp.workdps(256*8):
            for k in [1e3, 1e4, 1e5]:
                m1 = gsas_moment(2.0, alpha=alpha, k=k)
                m1a = 1 / _fcm_m2(alpha, k)
                m1b = _gsas_m2(alpha, k)
                delta_precise_up_to(m1a, m1b)
                delta_precise_up_to(m1, m1a, msg_prefix=f"alpha={alpha} k={k} ", abstol=0.2, reltol=0.005)

                m1 = gsas_moment(2.0, alpha=alpha, k=-k)
                m1a = 1 / _fcm_m2(alpha, -k)
                m1b = _gsas_m2(alpha, -k)
                delta_precise_up_to(m1a, m1b)
                delta_precise_up_to(m1, m1a, msg_prefix=f"alpha={alpha} k={-k} ", abstol=0.2, reltol=0.005)

    def test_var1(self): self._var(0.85)
    def test_var2(self): self._var(1.0)
    def test_var3(self): self._var(1.5)
    def test_var4(self): self._var(2.0)


# large k formula
class Test_ExKurt_at_Large_K:
    def translate_ex_kurt(self, ex_kurt, k):
        s = np.log(1.0 + ex_kurt/3) * (k-3)/4 + 0.5
        return gsas_kurtosis(1/s, k)
    
    def test_v1_at_10(self):
        k = 10.0
        ex_kurt = 1.0
        exact_ex_kurt = self.translate_ex_kurt(ex_kurt, k)
        delta_precise_up_to(ex_kurt, exact_ex_kurt, abstol=0.02, reltol=0.02)

    def test_v2_at_10(self):
        k = 10.0
        ex_kurt = 2.0
        exact_ex_kurt = self.translate_ex_kurt(ex_kurt, k)
        delta_precise_up_to(ex_kurt, exact_ex_kurt, abstol=0.04, reltol=0.02)

    def test_v3_at_20(self):
        k = 20.0
        ex_kurt = 0.5
        exact_ex_kurt = self.translate_ex_kurt(ex_kurt, k)
        delta_precise_up_to(ex_kurt, exact_ex_kurt, abstol=0.01, reltol=0.002)


# -------------------------------------------------------------------------------------
# 1/x expansion of PDF
def test_gsas_pdf_inv_x_series():
    alpha = 1.15
    k = 4.1
    x = 6.5
    
    p1 = gsas(alpha, k).pdf(x)
    p2 = GSaS_Wright(alpha, k, max_n=50).pdf_wright(x)
    delta_precise_up_to(p1, p2, abstol=1e-8)

    with mp.workdps(256*8):
        p3 = GSaS_Wright_MP(alpha, k, max_n=100).pdf(x)
        delta_precise_up_to(p1, float(p3), abstol=1e-8)


def test_gsas_pareto_tail_equiv():
    alpha = 1.15
    k = 1.0
    x = 6.5
    p1 = gsas(alpha, k).pdf(x)

    with mp.workdps(256*8):
        gmp = GSaS_Wright_MP(alpha, k, max_n=100)
        p2 = gmp.pdf(x)
        delta_precise_up_to(p1, float(p2), abstol=1e-8)
    
    # first term should match pareto tail
    p4 = gmp.pdf(x, show_terms=True)[0]
    c = gamma(alpha) * np.sin(alpha * np.pi/2) / np.pi
    p5 = c * alpha * x**(-(1+alpha))
    delta_precise_up_to(float(p4), p5, abstol=1e-8)


# -------------------------------------------------------------------------------------
class Test_GSaS_Series_X:
    # testing the equations in GSC MGF area, leading to GSaS Series in x
    alpha = 0.9
    k = 3.0

    gsc_alpha = alpha / 2
    p = gsc_alpha
    d = k/2
    sigma = fcm_sigma(alpha, k)
    gsc1 = gen_stable_count(alpha=gsc_alpha, sigma=sigma, d=d, p=p)
    gsc2 = gen_stable_count(alpha=gsc_alpha, sigma=sigma, d=2*d-1, p=2*p)

    def test_gsc_mgf(self):
        t = -0.05

        def fn1(x):  return self.gsc1.pdf(x) * np.exp(x*t)
        p1 = quad(fn1, a=0, b=np.inf, limit=100000)[0]

        p = self.p
        d = self.d
        sigma = self.sigma
        c = gamma(d) / gamma(d/p)
        p2 = c * Wright4Ways_MP(a=1/p, b=d/p, lam=1.0, mu=d).wright_fn(t*sigma, start=0, max_n=100) 

        delta_precise_up_to(p1, float(p2))

    def test_gsc_conversion(self):
        s = 0.55
        p = self.p
        d = self.d
        sigma = self.sigma

        g1 = self.gsc1.pdf(s**2/sigma)
        c = 0.5 * gamma(d) / gamma(d/p) / gamma(d-0.5) * gamma((d-0.5)/p)
        gsc2 = gen_stable_count(alpha=p, sigma=sigma, d=2*d-1, p=2*p)
        g2 = c * gsc2.pdf(s)

        delta_precise_up_to(g1, g2)

    def test_gsc_mgf_new(self):
        t = 0.5
        p = self.p
        d = self.d
        sigma = self.sigma

        def fn1(s): return s * np.exp(-(s*t)**2/2) * self.gsc1.pdf(s**2/sigma) * 2/sigma
        p1 = quad(fn1, a=0, b=np.inf, limit=100000)[0]
        c = gamma(d) / gamma(d/p)
        p2 = c * Wright4Ways_MP(a=1/p, b=d/p, lam=1.0, mu=d).wright_fn(-(t*sigma)**2/2, start=0, max_n=100) 

        delta_precise_up_to(p1, float(p2))

    def test_gsas_small_x(self):
        x = 0.5

        def fn2(s):  return s * self.gsc2.pdf(s) * norm().pdf(x*s)
        p1 = quad(fn2, a=0.000, b=np.inf, limit=100000)[0]
        p2 = gsas(self.alpha, self.k).pdf(x)
        delta_precise_up_to(p1, p2)

        with mp.workdps(256*8):
            gmp = GSaS_Wright_MP(self.alpha, self.k, max_n=100)
            p3 = gmp.pdf_by_small_x(x)

        delta_precise_up_to(p1, float(p3))
 
