
# test gsas

from re import X
import numpy as np
import pandas as pd
import mpmath as mp
from scipy.special import gamma
from scipy.integrate import quad
from scipy import stats
from scipy.stats import chi, chi2, norm

from .wright import wright_fn
from .frac_gamma import frac_gamma_star

from .stable_count_dist import gen_stable_count, gsc_normalization_constant, gsc_mu_by_m_series, sv_mu_by_f, gsc_pdf_large_x
from .gas_dist import gsas, lihn_stable, gsas_pdf_at_zero, gsas_moment
from .fcm_dist import fcm_moment, frac_chi_mean, FracChiMean, fcm_sigma,\
    fcm_mu_by_f, fcm_inverse_mu_by_f, fcm_q_by_f, fcm_q_by_fg_q, fcm_pdf_large_x, fcm_k1_mellin_transform,\
    frac_chi2_mean
from .unit_test_utils import *
from .hankel import *


# -------------------------------------------------------------------------------------
def _t_gsas(k): return gsas(alpha=1.0, k= k)
def _t_gas(k): return lihn_stable(alpha=1.0, k= k, theta=0.0)


# -------------------------------------------------------------------------------------
# t
def test_t_equiv():
    # P_1,k = t_k
    k = 3.0
    x = 0.85
    t_gsas = _t_gsas(k)
    compare_two_rvs(x, stats.t(k), t_gsas)

    t_gas = lihn_stable(alpha=1.0, k= k, theta=0.0)
    compare_two_rvs(x, stats.t(k), t_gas)


class Test_T_PdfAtZero:
    # t_k(0)
    k = 3.0
    p1 = stats.t(k).pdf(0.0)  # type: ignore

    def test_fn_form(self):
        p2 = gsas_pdf_at_zero(alpha=1.0, k=self.k)
        delta_precise_up_to(self.p1, p2)

    def test_gsas(self):
        p3 = _t_gsas(self.k).pdf(0.0)  # type: ignore
        delta_precise_up_to(self.p1, p3)
    
    def test_gas(self):
        p4 = _t_gas(self.k).pdf(0.0)  # type: ignore
        delta_precise_up_to(self.p1, p4)


def test_t_moments():
    for n in [2.0, 4.0]:
        for k in np.arange(n+1, 8.0):
            t_gsas = _t_gsas(k)
            m1 = stats.t(k).moment(n)
            m2 = t_gsas.moment(n)
            m3 = gsas_moment(n=n, alpha=1.0, k=k)
            delta_precise_up_to(m1, m2)
            delta_precise_up_to(m1, m3)


# -------------------------------------------------------------------------------------
# chi and FCM

class Test_Fractional_Chi1:
    # Section 4.1
    alpha = 0.75
    x = 0.85
    fcm = frac_chi_mean(alpha=alpha, k=1.0)
    gsc = gen_stable_count(alpha=alpha/2, sigma=1/np.sqrt(2), d=0.0, p=alpha)
    wr = 2/x * wright_fn(-(np.sqrt(2)*x)**alpha, -alpha/2, 0)  # type: ignore

    x_arr = np.linspace(0, 3.5, 100)

    def test_chi1_gsc_equiv(self):
        compare_two_rvs(self.x, self.fcm, self.gsc)

    def test_wright_equiv(self):
        delta_precise_up_to(self.wr, self.fcm.pdf(self.x))  # type: ignore

    def test_array_call(self):
        self.fcm.cdf(self.x_arr)
        self.fcm.pdf(self.x_arr)  # type: ignore


class Test_FCM_RVS:
    fcm = frac_chi_mean(alpha=1.2, k=6.0)
    x = fcm.rvs(200 * 1000)  # 10 seconds for 200k samples
     
    def test_rvs_mean(self):
        delta_precise_up_to(self.x.mean(), self.fcm.mean(), abstol=0.01, reltol=0.01)
    
    def test_rvs_var(self):
        delta_precise_up_to(self.x.var(), self.fcm.var(), abstol=0.05, reltol=0.01)


class Test_FCM_Chi_V1:
    # fcm_{1,k} = chi_k / sqrt(k) 
    k = 3.0
    x = 0.85
    fcm = frac_chi_mean(alpha=1.0, k=k)
    scaled_chi = chi(k, scale=1/np.sqrt(k))
    
    fcm2 = frac_chi2_mean(alpha=1.0, k=k)
    scaled_chi2 = chi2(k, scale=1/k)

    def test_equiv(self):
        compare_two_rvs(self.x, self.scaled_chi, self.fcm)

    def test_fcm_moment(self):
        for n in [1,2,3,4]:
            p1 = self.fcm.moment(float(n))
            p2 = self.scaled_chi.moment(float(n))
            delta_precise_up_to(p1, p2)

    def test_fcm_moment_fn(self):
        for n in [1,2,3,4]:
            p1 = fcm_moment(float(n), alpha=1.0, k=self.k)
            p2 = self.scaled_chi.moment(float(n))
            delta_precise_up_to(p1, p2)

    def test_fcm2_moment(self):
        for n in [1,2,3,4]:
            p1 = self.fcm2.moment(float(n))
            p2 = self.scaled_chi2.moment(float(n))
            delta_precise_up_to(p1, p2)


class Test_FCM_Chi_V2:
    # fcm_{1,k} = chi_k / sqrt(k) 
    k = 8.0
    x = 0.55
    fcm = frac_chi_mean(alpha=1.0, k=k)
    scaled_chi = chi(k, scale=1/np.sqrt(k))

    def test_equiv(self):
        compare_two_rvs(self.x, self.scaled_chi, self.fcm)
        
    def test_moments_equiv(self):
        for n in [1,2,3,4]:
            p1 = self.scaled_chi.moment(float(n))
            p2 = self.fcm.moment(float(n))
            delta_precise_up_to(p1, p2)


class Test_FCM_PDF:
    alpha = 0.85 
    k = 2.5
    f1 = frac_chi_mean(alpha=alpha, k=k)
    f2 = FracChiMean(alpha=alpha, k=k)
    
    def test_pdf(self):
        x = 0.75
        p1 = self.f1.pdf(x)  # type: ignore
        p2 = self.f2.pdf(x)  # type: ignore
        delta_precise_up_to(p1, p2)
        
    def test_pdf_by_wright_f(self):
        x = 0.65
        p1 = self.f1.pdf(x)  # type: ignore
        p2 = self.f2.pdf_by_wright_f(x)
        delta_precise_up_to(p1, p2)

    def test_pdf_by_mellin(self):
        x = 0.85
        p1 = self.f1.pdf(x)  # type: ignore
        p2 = self.f2.pdf_by_mellin(x)
        delta_precise_up_to(p1, p2)
 
 
# fcm moments
class Test_FCM_Moments:
    alpha = 0.75 
    k = 2.5
    fcm = frac_chi_mean(alpha=alpha, k=k)
    c_const = gsc_normalization_constant(alpha/2, sigma=1/np.sqrt(2*k), d=k-1, p=alpha)

    def mnt(self, n):
        return fcm_moment(n, alpha=self.alpha, k=self.k)

    def test_fcm_c_constant(self):
        k = self.k
        p1 = self.c_const 
        p2 = self.alpha * np.sqrt(2*k) * gamma((k-1)/2) / gamma((k-1)/self.alpha)
        delta_precise_up_to(p1, p2)
        
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
            delta_precise_up_to(m1, m2, msg_prefix=f"n={n} ")


def _fcm_mnt(n, alpha, k):
    alpha = mp.mpf(alpha)
    k = mp.mpf(k)
    m = alpha**(-n/alpha)
    return float(m)

class Test_FCM_Mean_LargeK:
    # Section 5.4
    def _mean(self, alpha): self._mnt(1.0, alpha)

    def _mnt(self, n, alpha):
        with mp.workdps(256*8):
            for k in [1e3, 1e4, 1e5]:
                m1 = fcm_moment(n, alpha=alpha, k=k)
                m1a = _fcm_mnt(n, alpha, k)
                delta_precise_up_to(m1, m1a, msg_prefix=f"alpha={alpha} k={k} n={n}", abstol=0.005, reltol=0.005)

                m1 = fcm_moment(n, alpha=alpha, k=-k)
                m1a = _fcm_mnt(-n, alpha, k)
                delta_precise_up_to(m1, m1a, msg_prefix=f"alpha={alpha} k={-k} n={n}", abstol=0.005, reltol=0.005)

    def test_mean1(self): self._mean(0.85)
    def test_mean2(self): self._mean(1.0)
    def test_mean3(self): self._mean(1.5)
    def test_mean4(self): self._mean(2.0)

    def test_more_mnt(self): 
        alpha = 0.75 
        self._mnt(2.0, alpha)
        self._mnt(3.0, alpha)


# fcm moments for negative k
class Test_FCM_Moments_NegK:
    alpha = 0.55  # the smaller alpha, the harder the integral is
    k = -3.5
    fcm = frac_chi_mean(alpha=alpha, k=k)

    def mnt(self, n):
        return fcm_moment(n, alpha=self.alpha, k=self.k)
    
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


class Test_FCM_ReflectionRule:
    alpha = 1.5
    k = 2.5

    f1 = frac_chi_mean(alpha, -k)
    f2 = frac_chi_mean(alpha, k)
    m1 = fcm_moment(1.0, alpha, k)

    def test_m1_closed_form(self):
        alpha = self.alpha
        k = self.k
        sigma = k**(0.5 - 1/alpha) / np.sqrt(2)
        m1a = sigma * gamma((k-1)/2) / gamma(k/2) * gamma(k/alpha) / gamma((k-1)/alpha)
        delta_precise_up_to(self.m1, m1a)

    def test_pdf_reflection(self):
        x = 1.5
        p1 = self.f1.pdf(x)  # type: ignore
        p2 = self.f2.pdf(1/x) / (x**3 * self.m1)  # type: ignore
        delta_precise_up_to(p1, p2)

    def test_moment_reflection(self):
        alpha = self.alpha
        k = self.k
        for n in [1,2,3]:  # n <= ceiling(k)
            p1 = fcm_moment(n, alpha, -k)
            p2 = fcm_moment(-n+1, alpha, k) / self.m1
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


# -------------------------------------------------------------------------------------
# this is slow
def test_fcm_hankel():
    alpha = 0.75 
    k = 2.5

    x = 0.75
    fcm = frac_chi_mean(alpha=alpha, k=k)
    p1 = fcm.pdf(x)  # type: ignore

    g = gamma((k-1)/2)
    sigma = fcm_sigma(alpha, k)
    
    def fcm_integrand(t):
        e_term = g * np.exp(t) / t**((k-1)/2)
        scale = sigma/np.sqrt(t)
        return e_term * pdf_gg(x, a=scale, d=k-1, p=alpha)

    q1 = hankel_integral_mpr(fcm_integrand)  # parallel version

    assert abs(np.imag(q1)) < 1e-8  # type: ignore
    delta_precise_up_to(p1, q1, reltol=1e-3)


# -------------------------------------------------------------------------------------
class Test_FCM_Mu:

    def test_mu_for_t(self):
        x = 0.55
        alpha = 1.0
        k = 2.3
        mu1 = fcm_mu_by_f(x, dz_ratio=0.0001, alpha=alpha, k=k)
        mu1a = fcm_mu_by_f(x, dz_ratio=None, alpha=alpha, k=k)
        delta_precise_up_to(mu1, mu1a)

        mu2 = k/2 * (1 - x**2)
        delta_precise_up_to(mu1, mu2)

        sigma = fcm_sigma(alpha, k)
        mu3 = gsc_mu_by_m_series(x, alpha=alpha/2, sigma=sigma, d=k-1.0, p=alpha)
        delta_precise_up_to(mu1, mu3)


    # use inverse velow
    def test_mu_for_exppow_sv_equiv(self):
        x = 0.55
        alpha = 1.2
        p1 = fcm_inverse_mu_by_f(x, dz_ratio=0.0001, alpha=alpha, k=-1.0)
        p2 = fcm_inverse_mu_by_f(x, dz_ratio=None, alpha=alpha, k=-1.0)
        delta_precise_up_to(p1, p2)

        p3 = sv_mu_by_f(x, dz_ratio=0.0001, alpha=alpha)
        p4 = sv_mu_by_f(x, dz_ratio=None, alpha=alpha)
        delta_precise_up_to(p3, p4)
        delta_precise_up_to(p1, p3)

    def test_fcm_q(self):
        x = 0.55
        alpha = 1.2
        p1 = fcm_q_by_f(x, dz_ratio=0.0001, alpha=alpha)
        p2 = fcm_q_by_fg_q(x, dz_ratio=0.0001, alpha=alpha)
        delta_precise_up_to(p1, p2)


class Test_FCM_Mu_Exp:
    x0 = np.sqrt(1.0/2)

    def _mu(self, x): return 1.0/(2*x**2) - 1.0
    
    def test_mu_for_exp_at_1(self):
        p1 = self._mu(1.0)
        p2 = fcm_mu_by_f(1.0, dz_ratio=None, alpha=1.0, k=-1.0)
        delta_precise_up_to(p1, p2)

    def test_mu_0_for_exp(self):
        p1 = 0.0  # x = 1/3, mu = 0
        p2 = fcm_mu_by_f(self.x0, dz_ratio=None, alpha=1.0, k=-1.0)
        delta_precise_up_to(p1, p2)

    def test_gsc_mu_for_exp_at_1(self):
        p1 = self._mu(1.0)
        p2 = gsc_mu_by_m_series(1.0, alpha=0.5, sigma=np.sqrt(2), d=-1.0, p=-1.0)
        delta_precise_up_to(p1, p2)

    def test_gsc_mu_0_for_exp(self):
        p1 = 0.0  # x = 1/3, mu = 0
        p2 = gsc_mu_by_m_series(self.x0, alpha=0.5, sigma=np.sqrt(2), d=-1.0, p=-1.0)
        delta_precise_up_to(p1, p2)

    def test_mu_inverse_for_exp(self):
        x = 0.55
        p1 = sv_mu_by_f(x, dz_ratio=None, alpha=1.0)
        p2 = fcm_inverse_mu_by_f(x, dz_ratio=None, alpha=1.0, k=-1.0)
        delta_precise_up_to(p1, p2)
        p3 = 1.0 - x**2/2
        delta_precise_up_to(p1, p3)


# ---------------------------------------------------------
# frac_gamma_star related tests
def test_fcm_cdf_vs_frac_gamma_star():
    alpha = 0.75 
    k = 2.5

    x = 0.75
    p1 = frac_chi_mean(alpha=alpha, k=k).cdf(x)
    p2 = FracChiMean(alpha=alpha, k=k).cdf_by_gamma_star(x)
    delta_precise_up_to(p1, p2)

