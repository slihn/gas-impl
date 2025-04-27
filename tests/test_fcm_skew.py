
# test gsas

import numpy as np
import pandas as pd
import mpmath as mp
from scipy.special import gamma
from scipy.integrate import quad
from scipy.stats import chi, norm

from .wright import wright_f_fn_by_levy

from .stable_count_dist import gen_stable_count, gsc_pdf_large_x
from .fcm_dist import fcm_moment, frac_chi_mean, FracChiMean, fcm_sigma, fcm_pdf_large_x, fcm_k1_mellin_transform, fcm_mellin_transform
from .gas_dist import g_from_theta
from .unit_test_utils import *
from .hankel import *
from .mellin import pdf_by_mellin


# ################################################## 
# Important: this file adds theta to all the tests #
# ################################################## 


# -------------------------------------------------------------------------------------
# chi and FCM
class Test_FCM_PDF_Positive_k:
    # Lihn(2025)
    alpha = 0.85
    theta = 0.40
    k = 1.5

    g = g_from_theta(alpha, theta)
    eps = 1/alpha 
    
    x = 0.85
    fcm = frac_chi_mean(alpha=alpha, k=k, theta=theta)
    sigma = fcm_sigma(alpha, k, theta)
    gsc = gen_stable_count(alpha=g*alpha, sigma=sigma, d=k-1, p=alpha)
    
    c =  gamma(g*(k-1)) /eps /gamma(eps*(k-1)) * sigma**(1-k) 
    wr = c * x**(k-2) * wright_f_fn_by_levy((x/sigma)**alpha, g*alpha)  # type: ignore

    def test_gsc_equiv(self):
        compare_two_rvs(self.x, self.fcm, self.gsc)

    def test_wright_equiv(self):
        delta_precise_up_to(self.wr, self.fcm.pdf(self.x))  # type: ignore

    def test_gsc_by_mellin(self):
        p1 = self.fcm.pdf(self.x)  # type: ignore
        c = (2-self.k) + 0.5
        p2 = pdf_by_mellin(self.x, lambda s: fcm_mellin_transform(s, self.eps, self.k, self.g), c=c)
        delta_precise_up_to(p1, p2)


class Test_FCM_PDF_Negative_k:
    # Lihn(2025)
    alpha = 0.85
    theta = 0.40
    k = 2.5

    g = g_from_theta(alpha, theta)
    eps = 1/alpha 
    
    x = 0.85
    fcm = frac_chi_mean(alpha=alpha, k=-k, theta=theta)
    sigma = fcm_sigma(alpha, k, theta)
    gsc = gen_stable_count(alpha=g*alpha, sigma=1/sigma, d=-k, p=-alpha)

    def test_gsc_equiv(self):
        compare_two_rvs(self.x, self.fcm, self.gsc)

    def test_gsc_by_mellin(self):
        p1 = self.fcm.pdf(self.x)  # type: ignore
        c = (self.k + 1.0) - 0.5
        p2 = pdf_by_mellin(self.x, lambda s: fcm_mellin_transform(s, self.eps, -self.k, self.g), c=c)
        delta_precise_up_to(p1, p2)


class Test_Fractional_Chi1:
    # Lihn(2025)
    alpha = 0.75
    theta = 0.35
    g = g_from_theta(alpha, theta)
    
    x = 0.85
    fcm = frac_chi_mean(alpha=alpha, k=1.0, theta=theta)
    gsc = gen_stable_count(alpha=g*alpha, sigma=g**g, d=0.0, p=alpha)
    wr = 1/(g*x) * wright_f_fn_by_levy(g**(-g*alpha) * x**alpha, g*alpha)  # type: ignore

    def test_chi1_gsc_equiv(self):
        compare_two_rvs(self.x, self.fcm, self.gsc)

    def test_wright_equiv(self):
        delta_precise_up_to(self.wr, self.fcm.pdf(self.x))  # type: ignore


# fcm moments
class Test_FCM_Moments:
    alpha = 0.75 
    theta = 0.15
    k = 2.5
    fcm = frac_chi_mean(alpha=alpha, k=k, theta=theta)

    def mnt(self, n):
        return fcm_moment(n, alpha=self.alpha, k=self.k, theta=self.theta)
        
    def test_moment_fn(self):
        for n in [1,2,3,4]:
            p1 = self.mnt(float(n))
            p2 = self.fcm.moment(float(n))
            delta_precise_up_to(p1, p2)

    def test_moments(self):
        for n in [1,2,3,4]:
            m1 = self.mnt(float(n))

            def fn2(x): return x**n * self.fcm.pdf(x)  # type: ignore
            m2 = quad(fn2, a=0.0001, b=np.inf, limit=100000)[0]
            delta_precise_up_to(m1, m2, msg_prefix=f"n={n} ", abstol=0.001, reltol=0.001)


# fcm moments for negative k
class Test_FCM_Moments_NegK:
    alpha = 0.55  # the smaller alpha, the harder the integral is
    theta = 0.15
    k = -3.5
    fcm = frac_chi_mean(alpha=alpha, k=k, theta=theta)

    def mnt(self, n):
        return fcm_moment(n, alpha=self.alpha, k=self.k, theta=self.theta)
    
    def test_moment_fn(self):
        # n can not be greater than abs(k)+1, when n+k is even, you get zero from Gamma(-(n+k)/2)
        for n in [1,2,3,4]:
            p1 = self.mnt(float(n))
            p2 = self.fcm.moment(float(n))
            delta_precise_up_to(p1, p2, msg_prefix=f"n={n} ")

    def test_moments(self):
        for n in [1,2,3]:  # n=4 integral is a bit problematic
            m1 = self.mnt(float(n))

            def fn2(x): return x**n * self.fcm.pdf(x)  # type: ignore
            m2 = quad(fn2, a=0.0001, b=np.inf, limit=100000)[0]
            delta_precise_up_to(m1, m2, msg_prefix=f"n={n} ", reltol=0.001)


# -------------------------------------------------------------------------------------
class Test_FCM_Mellin:
    alpha = 0.85
    k = 1.5
    theta = 0.2 

    fcm = FracChiMean(alpha, k, theta)
    fcm_k1 = FracChiMean(alpha, 1.0, theta)
    
    def test_pdf(self):
        for x in [0.45, 0.55]:
            for k in [2.5, -3.5]:
                fcm = FracChiMean(self.alpha, k, self.theta)
                p1 = fcm.pdf(x)
                p2 = fcm.pdf_by_mellin(x)
                delta_precise_up_to(p1, p2)

    def test_pdf_k1(self):
        for x in [0.45, 0.55]:
            p1 = self.fcm_k1.pdf(x)
            p2 = self.fcm_k1.pdf_by_mellin(x)
            delta_precise_up_to(p1, p2)
            
            eps = self.fcm_k1.eps
            g = self.fcm_k1.g
            c = (2-self.k) + 0.5  # c > 1

            p3 = pdf_by_mellin(x, lambda s: fcm_k1_mellin_transform(s, eps, g), c=c)
            delta_precise_up_to(p1, p3)


# -------------------------------------------------------------------------------------
def _fcm_asymp_mnt(n, alpha):
    alpha = mp.mpf(alpha)
    m = alpha**(-n/alpha)
    return float(m)


class Test_FCM_Mean_LargeK:
    def _mean(self, alpha): self._mnt(1.0, alpha)

    def _mnt(self, n, alpha):
        theta = 0.20
        with mp.workdps(256*8):
            for k in [1e3, 1e4, 1e5]:
                m1 = fcm_moment(n, alpha=alpha, k=k, theta=theta)
                m1a = _fcm_asymp_mnt(n, alpha)
                delta_precise_up_to(m1, m1a, msg_prefix=f"alpha={alpha} k={k} theta={theta} n={n}", abstol=0.005, reltol=0.005)

                m1 = fcm_moment(n, alpha=alpha, k=-k)
                m1a = _fcm_asymp_mnt(-n, alpha)
                delta_precise_up_to(m1, m1a, msg_prefix=f"alpha={alpha} k={-k} theta={theta} n={n}", abstol=0.005, reltol=0.005)

    def test_mean1(self): self._mean(0.85)
    def test_mean2(self): self._mean(1.0)
    def test_mean3(self): self._mean(1.5)
    def test_mean4(self): self._mean(2.0)

    def test_more_mnt(self): 
        alpha = 0.75 
        self._mnt(2.0, alpha)
        self._mnt(3.0, alpha)


class Test_FCM_ReflectionRule:
    alpha = 1.5
    theta = 0.25
    k = 2.5

    sigma = fcm_sigma(alpha, k, theta)

    f1 = frac_chi_mean(alpha, -k, theta)
    f2 = frac_chi_mean(alpha, k, theta)
    m1 = fcm_moment(1.0, alpha, k, theta)

    def test_m1_closed_form(self):
        alpha = self.alpha
        k = self.k
        g = g_from_theta(alpha, self.theta)
        m1a = self.sigma * gamma((k-1)*g) / gamma(k*g) * gamma(k/alpha) / gamma((k-1)/alpha)
        delta_precise_up_to(self.m1, m1a)

    def test_pdf_reflection(self):
        x = 1.5
        p1 = self.f1.pdf(x)  # type: ignore
        p2 = self.f2.pdf(1/x) / (x**3 * self.m1)  # type: ignore
        delta_precise_up_to(p1, p2)

    def test_moment_reflection(self):
        alpha = self.alpha
        k = self.k
        theta = self.theta
        for n in [1,2,3]:  # n <= ceiling(k)
            p1 = fcm_moment(n, alpha, -k, theta)
            p2 = fcm_moment(-n+1, alpha, k, theta) / self.m1
            delta_precise_up_to(p1, p2)


class Test_FCM_LargeX:
    def _locate_x(self, alpha, k):
        x = 1.0
        while True:
            assert x > 0.1
            p1 = frac_chi_mean(alpha=alpha, k=k).pdf(x)  # type: ignore
            if p1 < 1e-9: 
                x = x - 0.01
                continue
            if p1 < 1e-8: break
            x = x + 0.01
        return x 

    def test_fcm_large_x_case1(self):
        self._assert(alpha=0.85, k=4.5, msg="case1")

    def test_fcm_large_x_case2(self):
        self._assert(alpha=1.25, k=3.5, msg="case2")

    def test_fcm_large_x_case3(self):
        self._assert(alpha=1.5, k=2.5, msg="case3")

    def test_fcm_large_x_case4(self):
        x = 2.1
        k = 2.5 
        self._assert(alpha=1.0, k=k, msg="case4", tol=1e-5, x=x)

    def test_fcm_large_x_case5_chi_equiv(self):
        x = 0.5
        k = 3.5 

        p1 = fcm_pdf_large_x(x, 1.0, k)
        p2 = chi(k, scale=1/np.sqrt(k)).pdf(x)  # type: ignore
        delta_precise_up_to(p1, p2)

    def _assert(self, alpha, k, msg, tol=2e-3, x=None):
        if x is None:
            x = self._locate_x(alpha, k)
        p1 = frac_chi_mean(alpha=alpha, k=k).pdf(x)  # type: ignore
        p2 = gsc_pdf_large_x(x, alpha/2, fcm_sigma(alpha,k), d=k-1, p=alpha)
        p3 = fcm_pdf_large_x(x, alpha, k)

        assert p1 > 1e-10
        delta_precise_up_to(p1, p2, abstol=1e-2, reltol=tol, msg_prefix=msg + 'p1 vs p2')
        delta_precise_up_to(p1, p3, abstol=1e-2, reltol=tol, msg_prefix=msg + 'p1 vs p3')


