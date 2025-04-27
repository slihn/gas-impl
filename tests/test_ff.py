
import numpy as np
import pandas as pd
import mpmath as mp

from numpy import allclose  # type: ignore
from scipy.special import gamma
from scipy.integrate import quad
from scipy import stats
from scipy.stats import f, chi2, norm


from .fcm_dist import fcm_moment, frac_chi2_mean, chi2_11, chi2_11_moment
from .ff_dist import frac_f, FracF_PPF, frac_f_cdf_by_hyp2f1, frac_f_pdf_peak_at_zero, frac_f_cdf_by_gammainc
from .gas_dist import  gsas_squared, gsas
from .gas_sn_dist import  gsas_squared, gas_sn_squared
from .unit_test_utils import *
from .mellin import pdf_by_mellin


def test_chi2_eqv():
    d = 3.0
    x = 1.5 
    p1 = chi2(d, scale=1/d).pdf(x)  # type: ignore
    p2 = chi2(d).pdf(x*d) * d  # type: ignore
    p3 = frac_chi2_mean(alpha=1, k=d).pdf(x)  # type: ignore
    delta_precise_up_to(p1, p2)
    delta_precise_up_to(p1, p3)


# chi2_11 is tested here
class Test_Chi2_11:
    a = 2.2
    b = 6.6
    rho = 0.8
    c2 = chi2_11(a, b, rho=rho)

    size = 10*1000*1000
    z = c2.rvs(size)  # 10 million takes 15 seconds, it can get m1 to 2e-4, m2 to 2e-3

    def test_1st_moment(self):
        p1 = self.c2.mean()
        p2 = chi2_11_moment(1, self.a, self.b, self.rho)
        delta_precise_up_to(p1, p2)
        
        p3 = np.mean(self.z)
        delta_precise_up_to(p1, p3, abstol=1e-2, reltol=1e-3)

    def test_2nd_moment(self):
        p1 = self.c2.std()
        m1 = chi2_11_moment(1, self.a, self.b, self.rho)
        m2 = chi2_11_moment(2, self.a, self.b, self.rho)
        p2 = np.sqrt(m2 - m1**2)
        delta_precise_up_to(p1, p2)
        
        p3 = np.std(self.z)
        delta_precise_up_to(p1, p3, abstol=1e-2, reltol=5e-3)


# --------------------------------------------------------------------------
def test_ff_eqv_f():
    alpha = 1.0
    d = 3.0
    k = 4.0
    x = 1.0
    p1 = f(d, k).pdf(x)  # type: ignore
    p2 = frac_f(alpha, d, k).pdf(x)  # type: ignore
    delta_precise_up_to(p1, p2)


class Test_FF_GSaS_Square:
    alpha = 1.25
    d = 1.0
    k = 5.0

    rv1 = frac_f(alpha, d, k)
    rv2 = gsas_squared(alpha, k)
    rv3 = gas_sn_squared(alpha, k)  # this is the same as gsas_squared !

    def test_subsumes_gsas_squared(self):
        x = 1.5
        p1 = self.rv1.pdf(x)  # type: ignore
        p2 = self.rv2.pdf(x)  # type: ignore
        p3 = self.rv3.pdf(x)  # type: ignore

        delta_precise_up_to(p1, p2)
        delta_precise_up_to(p1, p3)

    def test_ff_ppdf(self):
        x = 1.5
        p1 = self.rv1.pdf(x)  # type: ignore
        rv9 = FracF_PPF(self.alpha, self.d, self.k)
        p9 = rv9.pdf(x) # type: ignore
        delta_precise_up_to(p1, p9)


def test_ff_2d_peak():
    alpha = 0.85
    k = 3.6
    d = 2.0
    p1 = frac_f(alpha, d=d, k=k).pdf(0.0)  # type: ignore
    p2 = fcm_moment(2.0, alpha=alpha, k=k)
    delta_precise_up_to(p1, p2)


def test_ff_peak():
    alpha = 0.85
    k = 3.6
    d = 1.9
    x = 0.0001
    p1 = frac_f(alpha, d=d, k=k).pdf(x)  # type: ignore
    p2 = frac_f_pdf_peak_at_zero(x, alpha=alpha, d=d, k=k)
    print(f"ff peak: p1={p1}, p2={p2}, err={p1/p2-1.0}")   # this is a bit problematic, so we want to see the error
    delta_precise_up_to(p1, p2, abstol=1e-2, reltol=1e-3)


class Test_FF_CDF:
    alpha = 1.2
    k = 4.6
    d = 2.1
    ff = frac_f(alpha, d=d, k=k)
    
    ff1 = frac_f(alpha, d=1.0, k=k)
    g = gsas(alpha, k)
    
    def test_ff_cdf(self):
        x = self.ff.mean()
        p1 = self.ff.cdf(x)  # type: ignore
        p2 = frac_f_cdf_by_gammainc(x, alpha=self.alpha, d=self.d, k=self.k)
        delta_precise_up_to(p1, p2)

    def test_ff_cdf_by_hyp2f1(self):
        x = self.ff.mean() * 1.1
        p1 = self.ff.cdf(x)  # type: ignore
        p2 = frac_f_cdf_by_hyp2f1(x, alpha=self.alpha, d=self.d, k=self.k)
        delta_precise_up_to(p1, p2)

    def test_ff_cdf_by_hyp2f1_integral(self):
        x = self.ff.mean() * 1.2
        p1 = self.ff.cdf(x)  # type: ignore
        p2 = frac_f_cdf_by_hyp2f1(x, alpha=self.alpha, d=self.d, k=self.k, integral=True)
        delta_precise_up_to(p1, p2)

    def test_ff_cdf_eq_gsas(self):
        x = self.ff1.mean() 
        p1 = self.ff1.cdf(x**2)  # type: ignore
        p2 = self.g.cdf(x) * 2 - 1.0 # type: ignore
        delta_precise_up_to(p1, p2)


class Test_FF_Moments:
    # very slow, takes 30 minutes
    alpha = 1.3
    k = 4.6
    ff1 = frac_f(alpha, d=1.0, k=k)
    ff2 = frac_f(alpha, d=2.0, k=k)

    ff_spy = frac_f(alpha=0.8, d=1.0, k=3.3)
    
    def test_ff1_moments(self):
        for n in [1, 2]:
            def _fn(x):  return x**n * self.ff1.pdf(x)  # type: ignore
            p1 = quad(_fn, 0, np.inf, limit=10000)[0]
            p2 = self.ff1.moment(n)  # type: ignore
            rtol = 1e-3 if n == 1 else 0.05
            print(f"ff1 moment: n={n}, p1={p1}, p2={p2}, err={p1/p2-1.0} vs rtol={rtol}")
            assert allclose(p1, p2, rtol=rtol)

    def test_ff2_moments(self):
        for n in [1, 2]:
            def _fn(x):  return x**n * self.ff2.pdf(x)  # type: ignore
            p1 = quad(_fn, 0, np.inf, limit=10000)[0]
            p2 = self.ff2.moment(n)  # type: ignore
            rtol = 1e-3 if n == 1 else 0.05
            print(f"ff2 moment: n={n}, p1={p1}, p2={p2}, err={p1/p2-1.0} vs rtol={rtol}")
            assert allclose(p1, p2, rtol=rtol)  # type: ignore

    def xx_test_ff_spy_moments(self):
        # TODO n=2 is hard, will take a look later
        for n in [1]: 
            def _fn(x):  return x**n * self.ff_spy.pdf(x)  # type: ignore
            p1 = quad(_fn, 0, np.inf, limit=10000)[0]
            p2 = self.ff_spy.moment(n)
            rtol = 0.05 if n == 1 else 0.1
            print(f"ff_spy moment: n={n}, p1={p1}, p2={p2}, err={p1/p2-1.0} vs rtol={rtol}")
            assert allclose(p1, p2, rtol=rtol)


# test Frac_F_Std_Adp_2D
# TODO