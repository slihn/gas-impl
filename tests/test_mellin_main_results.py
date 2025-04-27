
# test main results
# pyright: reportGeneralTypeIssues=false

import numpy as np
import pandas as pd

from scipy.special import gamma
from scipy.integrate import quad
from scipy import stats, special
from scipy.stats import gengamma, norm, levy_stable, chi, chi2, invgamma, invweibull

from .stable_count_dist import stable_count, gen_stable_count, gsc_normalization_constant, levy_stable_extremal, stable_vol
from .gas_dist import gsas, lihn_stable, frac_chi_mean,\
    g_skew_v2, g_skew_v2_at_s0, from_s1_to_feller, from_feller_to_s1,\
    gsas_pdf_at_zero, gsas_std_pdf_at_zero, gsas_moment, gsas_characteristic_fn
from .fcm_dist import frac_chi_mean, FracChiMean, fcm_moment, fcm_sigma, fcm_inverse_pdf, fcm_inverse
from .gexppow_dist import gexppow
from .wright import wright_fn, wright_f_fn
from .hankel import *
from .unit_test_utils import *


# FCM
class Test_Def_2_1:
    alpha = 0.85
    k = 3.8
    theta = 0.2

    fcm2 = FracChiMean(alpha, k, theta)  # this is more convenient for testing
    fcm = fcm2.fcm  # this is the gsc instance
    
    sigma = fcm_sigma(alpha, k, theta)
    gsc = gen_stable_count(alpha=alpha * fcm2.g,   sigma=sigma, d=k-1, p=alpha)
    C = gsc_normalization_constant(alpha * fcm2.g, sigma=sigma, d=k-1, p=alpha)

    fcm2_neg = FracChiMean(alpha, -k, theta)
    fcm_neg = fcm2_neg.fcm
    gsc_neg = gen_stable_count(alpha=alpha * fcm2.g, sigma=1/sigma, d=-k, p=-alpha)

    def test_fcm_vs_gsc(self):
        x = self.fcm.moment(1)
        p1 = self.fcm.pdf(x)
        p2 = self.gsc.pdf(x)
        delta_precise_up_to(p1, p2)

    def test_fcm_pdf_by_mellin_vs_gsc(self):
        x = self.fcm.moment(1)
        p1 = self.fcm2.pdf_by_mellin(x)
        p2 = self.gsc.pdf(x)
        delta_precise_up_to(p1, p2)

    def test_fcm_vs_wright(self):
        x = self.fcm.moment(1)
        p1 = self.fcm.pdf(x)

        alpha = self.alpha 
        k = self.k

        x1 = x / self.sigma
        y1 = x1**alpha
        p2 = self.C * x1**(k-2) * wright_fn(-y1, -alpha * self.fcm2.g, 0)
        delta_precise_up_to(p1, p2)

    def test_fcm_constant(self):
        alpha = self.alpha 
        k = self.k
        c1 = self.C * self.sigma
        c2 = alpha * gamma((k-1) * self.fcm2.g) / gamma((k-1)/alpha)
        delta_precise_up_to(c1, c2)

    def test_fcm_neg_vs_gsc(self):
        x = self.fcm_neg.moment(1)
        p1 = self.fcm_neg.pdf(x)
        p2 = self.gsc_neg.pdf(x)
        delta_precise_up_to(p1, p2)

    def test_fcm_neg_vs_fcm_pos_reflection(self):
        x = self.fcm_neg.moment(1)
        m1 = fcm_moment(1, self.alpha, self.k, self.theta)
        p1 = self.fcm_neg.pdf(x)
        p2 = self.fcm.pdf(1/x) / x**3 / m1
        delta_precise_up_to(p1, p2)

    def test_fcm_neg_moment_reflection(self):
        n = 3.0
        p1 = fcm_moment(n, self.alpha, -self.k, self.theta)
        p2 = fcm_moment(-n+1, self.alpha, self.k, self.theta) / fcm_moment(1, self.alpha, self.k, self.theta)
        delta_precise_up_to(p1, p2)

    def test_fcm_infinit_k(self):
        with mp.workdps(256*4):
            alpha = mp.mpf(self.alpha)
            eps = mp.mpf(self.fcm2.eps)
            g = mp.mpf(self.fcm2.g)
            k = 1e5

            m1 = fcm_moment(1.0, alpha=alpha, k=k, theta=self.theta)
            m2 = fcm_moment(2.0, alpha=alpha, k=k, theta=self.theta)
            p1 = m2 - m1**2
            delta_precise_up_to(p1, 0)

            p2 = float(mp.power(eps, eps))
            delta_precise_up_to(m1, p2, abstol=0.005, reltol=0.005)



# tests about chi_{alpha,k}
class Test_Def_3_Lemma_1:
    def test_alpha_0(self):
        alpha = 0.1
        x = 0.5
        chi = frac_chi_mean(alpha=alpha, k=1)
        levy = levy_stable(alpha=alpha, beta=0)
        p1 = levy.pdf(x)
        p2 = chi.pdf(x) / 2 
        p3 = alpha /(x * np.exp(1)) / 2
        delta_precise_up_to(p1, p2, abstol=0.001, reltol=0.01)
        delta_precise_up_to(p1, p3, abstol=0.005, reltol=0.01)


    def test_alpha_1(self):
        alpha = 1.0
        x = 0.35
        chi = frac_chi_mean(alpha=alpha, k=1)
        p1 = chi.pdf(x)
        p2 = norm().pdf(x) * 2
        delta_precise_up_to(p1, p2)

    def test_alpha_2(self):
        alpha = 1.98
        chi = frac_chi_mean(alpha=alpha, k=1)
        df2 = pd.DataFrame(data={'x': np.linspace(0.2, 1.2, num=401)})
        df2['p1']  = df2['x'].apply(lambda x: chi.pdf(float(x)))

        dx = df2.x.diff()[1]
        m1 = (df2.p1 * df2.x).sum() * dx
        m2 = (df2.p1 * df2.x**2).sum() * dx
        sd = (m2 - m1**2)**0.5

        m1_expected = 1/np.sqrt(2)  # 0.7071
        delta_precise_up_to(m1, m1_expected, abstol=0.01, reltol=0.01)
        assert sd < 0.05  # the width of the delta



class Test_Def_4:
    # GAS
    alpha = 0.85
    x = 0.35

    # test skewed gas at k=1
    def test_levy_stable_by_beta(self):
        # TODO this is a bit slow
        alpha = self.alpha
        beta  = 0.5

        theta, scale = from_s1_to_feller(alpha, beta)
        p1 = levy_stable(alpha=alpha, beta=beta, scale=scale)
        p2 = lihn_stable(alpha=alpha, k=1.0, theta=theta)
        compare_two_rvs(self.x, p1, p2)

    def test_levy_stable_by_theta(self):
        alpha = self.alpha
        theta = -0.71
        beta, scale = from_feller_to_s1(alpha, theta)
        p1 = levy_stable(alpha=alpha, beta=beta, scale=scale)
        p2 = lihn_stable(alpha=alpha, k=1.0, theta=theta)
        compare_two_rvs(self.x, p1, p2)


class Test_Def_4_Lemma_2_PDF:
    # GSaS
    alpha = 0.75
    x = 0.45

    def test_gsas_levy_stable(self):
        p1 = levy_stable(alpha=self.alpha, beta=0.0)
        p2 = gsas(alpha=self.alpha, k=1.0)
        compare_two_rvs(self.x, p1, p2)

        p3 = lihn_stable(alpha=self.alpha, k=1.0, theta=0.0)
        compare_two_rvs(self.x, p1, p3)

    def test_gsas_t(self):
        k = 2.5
        p1 = stats.t(k)
        p2 = gsas(alpha=1.0, k=k)
        compare_two_rvs(self.x, p1, p2)

        p3 = lihn_stable(alpha=1.0, k=k, theta=0.0)
        compare_two_rvs(self.x, p1, p3)



class Test_Def_5:
    # GEP
    # Def 5: GSaS and gexppow, via -k
    def test_gexppow_pdf_equal_gsas(self):
        x = 0.25
        for alpha in [0.75, 1.0, 1.25]:
            for k in [3.1, 4.3, 5.2]:
                p1 = gexppow(alpha=alpha, k=k).pdf(x)
                p2 = gsas(alpha=alpha, k=-k).pdf(x)
                delta_precise_up_to(p1, p2)



# test ratio vs product exchange
class Test_FCM_Inverse_GSaS:
    def test_gsas_pdf_ratio_vs_product(self):
        alpha = 0.8
        k = 4.8
        x = 0.25
        fc = frac_chi_mean(alpha=alpha, k=k)
        
        def fn1(s):  return s * norm().pdf(x*s) * fc.pdf(s) 
        def fn2(s):  return 1/s * norm().pdf(x/s) * fc.pdf(1/s) * s**(-2) 
        def fn3(s):  return 1/s * norm().pdf(x/s) * fcm_inverse_pdf(s, alpha, k) 
        def fn4(s):  return 1/s * norm().pdf(x/s) * fcm_inverse(alpha, k).pdf(s) 

        p1 = quad(fn1, a=0, b=50.0, limit=10000)[0] 
        p2 = quad(fn2, a=0, b=50.0, limit=10000)[0] 
        delta_precise_up_to(p1, p2)

        p3 = quad(fn3, a=0, b=50.0, limit=10000)[0] 
        delta_precise_up_to(p1, p3)

        p4 = quad(fn4, a=0, b=50.0, limit=10000)[0] 
        delta_precise_up_to(p1, p4)

    def test_gsas_cf_ratio_vs_product(self):
        alpha = 0.8
        k = 4.8
        x = 0.25
        fc = frac_chi_mean(alpha=alpha, k=k)
        
        c = np.sqrt(2*np.pi)
        def norm_cf(x): return c * norm().pdf(x)
        
        def fn1(s):  return norm_cf(x/s) * fc.pdf(s) 
        def fn2(s):  return s * norm_cf(x*s) * fc.pdf(1/s) * s**(-3) 

        p1 = quad(fn1, a=0, b=50.0, limit=10000)[0] 
        p2 = quad(fn2, a=0, b=50.0, limit=10000)[0] 
        delta_precise_up_to(p1, p2)

    def test_fcm_inverse_at_positive_k(self):
        alpha = 0.8
        k = 4.8
        x = 0.25
        p1 = fcm_inverse_pdf(x, alpha, k) 
        p2 = fcm_inverse(alpha, k).pdf(x)
        delta_precise_up_to(p1, p2)


class Test_FCM_Inverse_GEP:
    def test_fcm_inverse_at_negative_k(self):
        alpha = 0.8
        k = -4.8
        x = 0.25
        p1 = fcm_inverse_pdf(x, alpha, k) 
        p2 = fcm_inverse(alpha, k).pdf(x)
        delta_precise_up_to(p1, p2)

    def test_gep_with_inverse_pdf(self):
        alpha = 0.8
        k = 4.8
        x = 0.25
        p1 = gexppow(alpha, k).pdf(x)
        
        def fn2(s):  return norm().pdf(x/s) / s * fcm_inverse_pdf(s, alpha, -k)
        p2 = quad(fn2, a=0, b=50.0, limit=10000)[0] 
        delta_precise_up_to(p1, p2)

    def test_gep_with_inverse(self):
        alpha = 0.8
        k = 4.8
        x = 0.25
        p1 = gexppow(alpha, k).pdf(x)
        
        def fn2(s):  return norm().pdf(x/s) / s * fcm_inverse(alpha, -k).pdf(s)
        p2 = quad(fn2, a=0, b=50.0, limit=10000)[0] 
        delta_precise_up_to(p1, p2)

    def test_fcm_inverse_equal_stable_vol(self):
        alpha = 0.8
        x = 0.75
        p1 = fcm_inverse(alpha, k=-1.0).pdf(x)
        p2 = stable_vol(alpha).pdf(x)
        delta_precise_up_to(p1, p2)
