
# test gsas

from re import X
import numpy as np
import pandas as pd
import mpmath as mp
from scipy.special import gamma
from scipy.integrate import quad
from scipy import stats
from scipy.stats import chi, chi2, norm


from .stable_count_dist import gen_stable_count
from .fcm_dist import fcm_moment, frac_chi2_mean, FCM2, fcm2_moment, fcm2_hat, fcm2_hat_mellin_transform
from .unit_test_utils import *
from .hankel import *
from .mellin import pdf_by_mellin


# ---------------------------------------------------------
# ---------------------------------------------------------
class Test_FCM2:
    alpha = 0.75 
    k = 2.5

    fcm2 = frac_chi2_mean(alpha=alpha, k=k)
    fcm2a = FCM2(alpha=alpha, k=k)

    fcm2_neg = frac_chi2_mean(alpha=alpha, k=-k)
    fcm2a_neg = FCM2(alpha=alpha, k=-k)

    def test_fcm2_pdf(self):
        x = 0.75
        p1 = self.fcm2.pdf(x)  # type: ignore
        p2 = self.fcm2a.pdf(x)
        delta_precise_up_to(p1, p2)
        
    def test_fcm2_pdf_neg(self):
        X = 0.75
        p1 = self.fcm2_neg.pdf(X)  # type: ignore
        p2 = self.fcm2a_neg.pdf(X)
        delta_precise_up_to(p1, p2)

    def test_fcm2_pdf_by_fcm(self):
        x = 0.65
        p1 = self.fcm2.pdf(x)  # type: ignore
        p2 = self.fcm2a.pdf_by_fcm(x)
        delta_precise_up_to(p1, p2)

    def test_fcm2_pdf_by_mellin(self):
        x = 0.65
        p1 = self.fcm2.pdf(x)  # type: ignore
        p2 = self.fcm2a.pdf_by_mellin(x)
        delta_precise_up_to(p1, p2)

    def test_fcm2_pdf_by_wright_f(self):
        x = 0.65
        p1 = self.fcm2.pdf(x)  # type: ignore
        p2 = self.fcm2a.pdf_by_wright_f(x)
        delta_precise_up_to(p1, p2)

    def test_fcm2_cdf_vs_frac_gamma_star(self):
        x = 0.75
        p1 = self.fcm2.cdf(x)  # type: ignore
        p2 = self.fcm2a.cdf_by_gamma_star(x)        
        delta_precise_up_to(p1, p2)

    def test_fcm2_moments(self):
        for n in [1, 2, 3]:
            p1 = self.fcm2.moment(n)  # type: ignore
            p2 = fcm_moment(2*n, self.alpha, self.k)
            delta_precise_up_to(p1, p2)
    
    def test_fcm2_neg_moments(self):
        for n in [-1, -2]:
            p1 = fcm2_moment(n, self.alpha, self.k)
            p3 = self.fcm2a.moment_by_mellin(n)
            delta_precise_up_to(p1, p3)

    def test_fcm2_cdf_by_mellin(self):
        x = 0.65
        p1 = self.fcm2.cdf(x)  # type: ignore
        p2 = self.fcm2a.cdf_by_mellin(x)
        delta_precise_up_to(p1, p2)

    def test_fcm2_pdf_multiplied_by_x_m(self):
        x = 0.65
        m = 1.5
        p1 = self.fcm2a.pdf(x) * x**m
        p2 = self.fcm2a.pdf_multiplied_by(x, m)
        delta_precise_up_to(p1, p2)

# ----------------------------------------------------------------------------
class Test_FCM2_Hat:
    alpha = 1.0
    k = 2.5
    fcm2h = fcm2_hat(alpha=alpha, k=k)

    def test_fcm2_hat_reduce(self):
        s = 1.5
        p1 = fcm2_hat_mellin_transform(s, self.alpha, self.k)  # type: ignore
        k2 = self.k / 2.0
        p2 = gamma(s + k2 -1) / gamma(k2)
        delta_precise_up_to(p1, p2)  # type: ignore

    def test_fcm2_hat_pdf_by_mellin(self):
        x = 1.5
        p1 = self.fcm2h.pdf(x) # type: ignore
        c = 2.0 - self.k + 0.5
        p2 = pdf_by_mellin(x, lambda s: fcm2_hat_mellin_transform(s, self.alpha, self.k), c=c)
        delta_precise_up_to(p1, p2)  # type: ignore
