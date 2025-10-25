
# test gsc and sc

import numpy as np
import pandas as pd

from scipy.special import gamma, hyp1f1, poch
from scipy.integrate import quad
from scipy.stats import norm, levy_stable

from .stable_count_dist import wright_f_fn_by_sc, gsc_q_by_f, mainardi_wright_fn_in_gsc
from .wright import *
from .wright_asymp import *
from .mellin import pdf_by_mellin
from .unit_test_utils import *


# ----------------------------------------------------------------
def test_one_sided_vs_wright():
    alpha = 0.35
    levy1 = levy_stable_extremal(alpha)
    x = 0.85
    p = levy1.pdf(x)  # type: ignore

    q = wright_fn(-x**(-alpha), -alpha, 0) / x  # type: ignore
    delta_precise_up_to(p, q)

    q2 = wright_f_fn(x**(-alpha), alpha) / x  # type: ignore
    delta_precise_up_to(p, q2)

    q3 = mainardi_wright_fn(x**(-alpha), alpha) * alpha * x**(-alpha-1)  # type: ignore
    delta_precise_up_to(p, q3)


class Test_Wright_F_Fn:
    def test_f_wright_levy(self):
        alpha = 0.45
        x = 0.85
        p = wright_f_fn(x, alpha) 
        q = wright_f_fn_by_levy(x, alpha)
        delta_precise_up_to(p, q)

    def test_log_f_wright_levy(self):
        alpha = 0.45
        x = 0.85
        p = wright_f_fn_by_levy(x, alpha) 
        q = np.exp(log_wright_f_fn_by_levy(np.log(x), alpha))
        delta_precise_up_to(p, q)


# ----------------------------------------------------------------
# M-Wright, supposed to be very easy to converge

def test_m_wright_vs_exp():
    x = 0.85
    p1 = np.exp(-x)
    p2 = mainardi_wright_fn(x, 0.0) 
    delta_precise_up_to(p1, p2)

    p3 = wright_m_fn_by_levy(x, 0.0) 
    delta_precise_up_to(p1, p3)


def test_m_wright_at_zero_for_all_alphas():
    alpha = np.linspace(0.0, 1.0, 100)
    p1 = 1.0/gamma(1-alpha)
    p2 = np.array([wright_m_fn_by_levy(0.0, a1) for a1 in alpha])  # type: ignore
    rms = np.sqrt(np.mean((p1 - p2)**2))
    assert rms < 1e-6, f"rms = {rms} is too large"


def test_m_wright_vs_norm():
    x = 0.85
    p = mainardi_wright_fn(x, 0.5) 
    q = np.exp(-x**2/4) / np.sqrt(np.pi)
    delta_precise_up_to(p, q)

    q2 = norm(scale=np.sqrt(2)).pdf(x) * 2  # type: ignore  # half-normal
    delta_precise_up_to(p, q2)


def test_m_wright_vs_levy2_0():
    # (A37) of Mainardi (2020)
    levy2 = levy_stable(alpha=2.0, beta=0)
    x = 0.85
    p = levy2.pdf(x)  # type: ignore

    q = mainardi_wright_fn(x, 0.5) / 2.0  # type: ignore
    delta_precise_up_to(p, q)

    q2 = norm(scale=np.sqrt(2)).pdf(x)  # type: ignore  # normal, variance = 2
    delta_precise_up_to(p, q2)


def test_m_wright_for_small_x_and_alpha():
    for alpha in [0.0, 0.001, 0.002, 0.01, 0.02, 0.1]:
        for x in [0.0, 0.001, 0.002, 0.01, 0.02, 0.1]:
            p1 = mainardi_wright_fn(x, alpha, max_n=12)
            p2 = wright_m_fn_by_levy(x, alpha)
            delta_precise_up_to(p1, p2, msg_prefix=f"alpha = {alpha}, x = {x}: ")


class Test_Wright_Mellin:
    g = 0.45
    x = 0.35
    p_norm = norm.pdf(x)
    
    def eval_mellin(self, fn, mellin_fn):
        p1 = fn(self.x, self.g)  
        p2 = pdf_by_mellin(self.x, lambda s: mellin_fn(s, self.g))
        return (p1, p2)
    
    def test_wright_fn(self):
        lam = -0.45
        mu = 0.1
        x = 0.35
        p1 = wright_fn(-x, lam, mu)
        p2 = pdf_by_mellin(x, lambda s: wright_fn_mellin_transform(s, lam, mu))
        delta_precise_up_to(p1, p2)

    def test_wright_ratio_fn(self):
        lam = -0.45
        mu = 0.1
        x = -0.35
        p1 = wright_ratio_fn(x, lam, mu, lam)
        p2 = wright_fn(x, lam, mu+lam) / wright_fn(x, lam, mu)  # type: ignore
        delta_precise_up_to(p1, p2)

    def test_wright_ratio_fn_list(self):
        lam = -0.46
        mu = 0.1
        x = np.array([-0.35, -0.4, -0.45])
        p1 = wright_ratio_fn(x, lam, mu, lam)
        p2 = wright_fn(x, lam, mu+lam) / wright_fn(x, lam, mu)  # type: ignore
        q = p1 - p2
        assert np.all(np.abs(q) < 1e-4), f"max diff = {np.max(np.abs(q))} is too large"

    def test_wright_ratio_fn_recurrence(self):
        lam = -0.44
        mu = 0.1
        x = -0.34
        p1 = lam * x * wright_ratio_fn(x, lam, mu, lam)  # type: ignore
        p2 = wright_ratio_fn(x, lam, mu, delta=-1.0) + (1-mu)  # type: ignore
        delta_precise_up_to(p1, p2)

    def test_wright_f(self):
        p1, p2 = self.eval_mellin(wright_f_fn_by_levy, wright_f_fn_mellin_transform)  
        delta_precise_up_to(p1, p2)
        
    def test_wright_m(self):
        p1, p2 = self.eval_mellin(wright_m_fn_by_levy, wright_m_fn_mellin_transform)  
        delta_precise_up_to(p1, p2)

    def test_wright_m_rescaled(self):
        p1, p2 = self.eval_mellin(wright_m_fn_rescaled_by_levy, wright_m_fn_rescaled_mellin_transform)  
        delta_precise_up_to(p1, p2)

    def test_norm(self):
        p2 = pdf_by_mellin(self.x, norm_mellin_transform)
        delta_precise_up_to(self.p_norm, p2)
        
    def test_norm_v2(self):
        def norm_mellin_transform_v2(s):
            g0 = 0.5
            return g0**((s+1.0)/2) * gamma(s) / gamma(g0 * (s+1))

        p3 = pdf_by_mellin(self.x, norm_mellin_transform_v2)
        delta_precise_up_to(self.p_norm, p3)

    def test_norm_vs_wright_m_ercaled(self):
        p3 = pdf_by_mellin(self.x, lambda s: wright_m_fn_rescaled_mellin_transform(s, g=0.5))
        delta_precise_up_to(self.p_norm, p3)

    def test_m_wright_moments(self):
        for n in [0.0, 1.0, 2.0]:
            p1 = wright_m_fn_moment(n, self.g)
            p2 = quad(lambda x: x**n * wright_m_fn_by_levy(x, self.g), a=0, b=np.inf, limit=100000)[0]  # type: ignore
            delta_precise_up_to(p1, p2, msg_prefix=f"n = {n}: by formula")

    def test_m_wright_mean_variance(self):
        p1 = wright_m_fn_mean(self.g)
        p2 = 1/gamma(self.g + 1)
        delta_precise_up_to(p1, p2, msg_prefix=f"mean: by formula")

        p1 = wright_m_fn_std(self.g)**2
        p2 = 2/gamma(2*self.g + 1) - 1/gamma(self.g + 1)**2
        delta_precise_up_to(p1, p2, msg_prefix=f"variance: by formula")


class Test_M_Wright_Elasticty:
    alpha = 0.7
    x = 0.5
    p1 = wright_m_fn_elasticity_by_levy(x, alpha)
    
    def test_series_ratio(self): 
        p2 = wright_m_fn_elasticity_by_series(self.x, self.alpha)
        delta_precise_up_to(self.p1, p2)
    
    def test_versions(self):
        for version in [1,2,3]:
            p2 = wright_fn_elasticity(-self.x, -self.alpha, 1-self.alpha, version=version)
            delta_precise_up_to(self.p1, p2, msg_prefix=f"version={version}: ")

    def test_series_sum(self):
        p2 = wright_m_fn_elasticity(self.x, self.alpha)
        delta_precise_up_to(self.p1, p2)

    def test_one_half(self):
        x = 0.7  # x needs to be a bit larger
        p1 = -0.5 * x**2
        p2 = wright_m_fn_elasticity(x, alpha=0.5)
        delta_precise_up_to(p1, p2)

        p3 = wright_m_fn_elasticity_by_series(x, alpha=0.5)
        delta_precise_up_to(p1, p3)

        p4 = wright_m_fn_elasticity_by_levy(x, alpha=0.5)
        delta_precise_up_to(p1, p4)

    def test_relation_to_f(self):
        p2 = wright_f_fn_elasticity_by_levy(self.x, self.alpha)
        delta_precise_up_to(self.p1 + 1.0, p2)  # type: ignore

    def test_relation_to_q(self):
        p1 = self.p1 * self.alpha + 1.0 + self.alpha  # type: ignore
        p2 = wright_q_fn(self.x, self.alpha)
        delta_precise_up_to(p1, p2)


# ----------------------------------------------------------------
class Test_M_Wright_Variants():
    alpha = 0.65  # > 0.5
    x1 = 0.45
    x2 = -0.35

    p1 = wright_m_fn_by_levy(x1, alpha) 
    p2 = wright_m_fn_by_levy(x2, alpha) 
    
    def test_mainardi_vs_levy(self):
        q1 = mainardi_wright_fn(self.x1, self.alpha)
        delta_precise_up_to(self.p1, q1)

    def test_levy_vs_ts_x1(self):
        q1 = wright_mainardi_fn_ts(self.x1, self.alpha) 
        delta_precise_up_to(self.p1, q1)
        
    def test_levy_vs_ts_x2(self):
        q2 = wright_mainardi_fn_ts(self.x2, self.alpha) 
        delta_precise_up_to(self.p2, q2)

    def test_levy_vs_ts_many(self):
        alpha_list = [ 0.8, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99 ]
        x_list = [ 0.01, 0.1, 0.2, 0.5, 0.75, 0.9, 0.99, 1.0 ]
        for alpha in alpha_list:
            for x in x_list:
                p1 = wright_mainardi_fn_ts(x, alpha)
                p2 = wright_m_fn_by_levy(x, alpha)
                delta_precise_up_to(p1, p2, msg_prefix=f"alpha={alpha}, x={x}: ")


class Test_M_Wright_CDF():
    def test_cdf_impl_ts_vs_series(self):
        for alpha in [0.3, 0.5, 0.7]:
            for x in [0.0, 0.1, 0.5, 1.0]:
                p1 = wright_mainardi_fn_cdf_ts(x, alpha)
                p2 = mainardi_wright_fn_cdf(x, alpha)
                delta_precise_up_to(p1, p2, msg_prefix=f"alpha = {alpha}, x = {x}: ")

    def test_cdf_levy_vs_series(self):
        for alpha in [0.5, 0.7, 0.8]:
            for x in [0.0, 0.1, 0.5, 1.0]:
                p1 = mainardi_wright_fn_cdf_by_levy(x, alpha)
                p2 = mainardi_wright_fn_cdf(x, alpha)
                delta_precise_up_to(p1, p2, msg_prefix=f"alpha = {alpha}, x = {x}: ")

    def test_cdf_levy_vs_ts(self):
        x = 1.0
        for alpha in [0.5, 0.7, 0.9]:
            p3 = wright_mainardi_fn_cdf_ts(x, alpha)
            p4 = mainardi_wright_fn_cdf_by_levy(x, alpha)
            delta_precise_up_to(p3, p4, msg_prefix=f"alpha = {alpha}, x = {x}: ")

    def test_cdf_large_x(self):
        x = 100.0
        for alpha in [0.1, 0.5, 0.7, 0.9]:
            if alpha <= 0.5:
                p1 = wright_mainardi_fn_cdf_ts(x, alpha)
                delta_precise_up_to(p1, 1.0, msg_prefix=f"alpha = {alpha}, x = {x}: cdf_ts=1.0")
            if alpha >= 0.5:
                p2 = mainardi_wright_fn_cdf_by_levy(x, alpha)
                delta_precise_up_to(p2, 1.0, msg_prefix=f"alpha = {alpha}, x = {x}: cdf_levy=1.0")

    def test_cdf_series_small_alpha_x(self):
        for alpha in [0.0, 0.01, 0.05, 0.1]:
            for x in [0.0, 0.01, 0.02]:
                p1 = mainardi_wright_fn_cdf(x, alpha)
                p2 = -1* sum([ (-x)**n / gamma(n) / gamma(-alpha*n + 1.0) for n in range(1,5) ])
                assert p1 >= 0.0
                delta_precise_up_to(p1, p2, msg_prefix=f"alpha = {alpha}, x = {x}: ", abstol=1e-2, reltol=1e-2)
    
    def test_cdf_class_vs_gsc(self):
        for alpha in [0.1, 0.4, 0.5, 0.7, 0.9]:
            for x in [0.0, 0.01, 0.02, 0.1, 0.5, 1.0, 10.0, 50.0, 100.0, 200.0, np.inf]:
                m_wr1 = mainardi_wright_fn_in_gsc(alpha)
                m_wr2 = M_Wright_One_Sided(alpha)
                p1 = m_wr1.cdf(x)
                p2 = m_wr2.cdf(x)
                delta_precise_up_to(p1, p2, msg_prefix=f"alpha = {alpha}, x = {x}: ")


def test_poch():
    n = 5
    s = 1.6
    p = 1.0/(n+s)
    q = poch(s, n) / poch(s+1, n) / s
    delta_precise_up_to(p, q)


def test_m_wright_moments():
    alpha = 0.47
    m_wr2 = M_Wright_One_Sided(alpha)

    for n in [0.0, 1.0, 2.0]:
        def fn(x): return x**n * mainardi_wright_fn(x, alpha, max_n=80)  # max_n depends on max(x)
    
        p = quad(fn, a=0, b=8.0, limit=100000)[0]  # b can not be infinitely large, unfortunately
        q = gamma(n+1) / gamma(n*alpha + 1)
        delta_precise_up_to(p, q, abstol=0.001, reltol=0.001, msg_prefix=f"n = {n}: by formula")

        q2 = m_wr2.moment(n)
        delta_precise_up_to(p, q2, abstol=0.001, reltol=0.001, msg_prefix=f"n = {n}: by class")


def test_m_wright_diff():
    alpha = 0.48
    z = 0.5
    dz = z * 0.0001

    f = wright_f_fn_by_sc(z, alpha)
    f_dz =  wright_f_fn_by_sc(z+dz, alpha)
    df_dz = (f_dz - f)/dz

    m = mainardi_wright_fn(z, alpha)
    m_dz = mainardi_wright_fn(z+dz, alpha)
    dm_dz = (m_dz - m)/dz # type: ignore
    wr = mainardi_wright_fn_slope(z, alpha)
    delta_precise_up_to(dm_dz, wr)

    df_dz2 = alpha * m + alpha * z * dm_dz  # type: ignore
    delta_precise_up_to(df_dz, df_dz2)


class Test_Skew_MWright_Dist:
    g = 0.65
    dist = Skew_MWright_Dist(g)
    norm_dist = Skew_MWright_Dist(0.5)
    
    def test_pdf(self):
        for x in [-0.7, -0.1, 0.0, 0.1, 0.85]:
            p1 = self.dist.pdf(x)
            p2 = self.dist.pdf_by_fn(x)
            delta_precise_up_to(p1, p2, msg_prefix=f"x = {x}: pdf failed")

    def test_cdf(self):
        x = -40.0
        p1 = self.dist.cdf(x)
        delta_precise_up_to(p1, 0, abstol=1e-3, msg_prefix=f"x = {x}: cdf failed")

        x = 20.0
        p2 = self.dist.cdf(x)
        delta_precise_up_to(p2, 1.0, abstol=1e-3, reltol=1e-3, msg_prefix=f"x = {x}: cdf failed")

    def test_norm(self):
        for x in [-0.7, -0.1, 0.0, 0.1, 0.85]:
            p1 = self.norm_dist.pdf(x)
            p2 = norm().pdf(x)  # type: ignore
            delta_precise_up_to(p1, p2, msg_prefix=f"x = {x}: pdf vs norm failed")

    def test_total_density(self):
        c = quad_total_density(self.dist.pdf)
        delta_precise_up_to(c, 1.0)
    

def test_mu_wright_ratio():
    # this is the RV lemma
    alpha = 0.48
    z = 0.5

    f = wright_f_fn_by_sc(z, alpha)
    wr = wright_fn(-z, -alpha, -1.0)
    p1 = -wr/f  # type: ignore

    dz = z * 0.0001
    f_dz =  wright_f_fn_by_sc(z+dz, alpha)
    p2 = alpha * z * (f_dz - f)/dz / f + 1

    m = mainardi_wright_fn(z, alpha)
    m_dz = mainardi_wright_fn(z+dz, alpha)
    p3 = alpha * z * (m_dz - m)/dz / m + (alpha + 1)  # type: ignore

    delta_precise_up_to(p1, p2)
    delta_precise_up_to(p2, p3)

def test_mu_wright_ratio_by_q():
    # this is the RV lemma, by Q
    alpha = 0.48
    z = 0.45

    p1 = gsc_q_by_f(z, dz_ratio=0.001, alpha=alpha)
    p2 = wright_q_fn(z, alpha)
    delta_precise_up_to(p1, p2)
    
    
def test_m_wright_slope_at_zero():
    for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
        p1 = mainardi_wright_fn_slope(0.0, alpha)
        p2 = -1/np.pi * gamma(2*alpha) * np.sin(2*alpha*np.pi) 
        delta_precise_up_to(p1, p2, msg_prefix=f"alpha = {alpha}: ")

        dz = 0.0001
        m = mainardi_wright_fn(0.0, alpha)
        m_dz = mainardi_wright_fn(dz, alpha)
        p3 = (m_dz - m)/dz  # type: ignore
        delta_precise_up_to(p1, p3, abstol=0.01, reltol=0.001, msg_prefix=f"dm_dz alpha = {alpha}: ")