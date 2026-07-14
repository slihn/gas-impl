
# test gsas

import imp
import numpy as np
import pandas as pd
import pytest
from scipy.special import gamma
from scipy.integrate import quad
from scipy import stats
from scipy.stats import norm, cauchy, levy_stable, chi

from .stable_count_dist import stable_count, gen_stable_count
from .gas_dist import gsas, lihn_stable, LihnStable,\
    g_skew_v2, g_skew_v2_osc, gsas_pdf_at_zero, gsas_std_pdf_at_zero, gsas_moment,\
    from_feller_to_s1, from_s1_to_feller, levy_stable_from_feller
from .fcm_dist import fcm_k1_mellin_transform
from .wright import wright_m_fn_rescaled_mellin_transform, wright_m_fn_rescaled_by_levy
from .mellin import pdf_by_mellin
from .unit_test_utils import *


# -------------------------------------------------------------------------------------
def _cauchy(x): return cauchy().pdf(x)  # type: ignore
def _norm(x): return norm().pdf(x)  # type: ignore


# -------------------------------------------------------------------------------------
# basic stuff


def test_sas_equiv():
    # P_alpha_k = P_alpha
    alpha = 0.55
    x = 0.85
    sas = levy_stable(alpha=alpha, beta=0)
    sas_gsas = gsas(alpha=alpha, k=1.0)
    compare_two_rvs(x, sas, sas_gsas, min_p=0.1)

    sas_gas = lihn_stable(alpha=alpha, k=1.0, theta=0.0)
    compare_two_rvs(x, sas, sas_gas, min_p=0.1)


class Test_Feller_Beta_Conversion:
    @staticmethod
    def expected_scale(alpha, theta):
        return np.cos(theta * np.pi/2) ** (1.0/alpha)

    def test_feller_to_s1_at_zero(self):
        for alpha in [0.01, 0.8, 1.0, 1.2, 1.99, 2.0]:
            beta, scale = from_feller_to_s1(alpha, theta=0.0)
            np.testing.assert_allclose([beta, scale], [0.0, 1.0])

    def test_feller_to_s1_at_theta_boundaries(self):
        # Below alpha=1 the signs of theta and beta are opposite. Above
        # alpha=1 they agree, and the theta boundary is 2-alpha.
        cases = [
            (0.01, -0.01, 1.0),
            (0.01, 0.01, -1.0),
            (0.2, -0.2, 1.0),
            (0.2, 0.2, -1.0),
            (0.8, -0.8, 1.0),
            (0.8, 0.8, -1.0),
            (1.2, -0.8, -1.0),
            (1.2, 0.8, 1.0),
            (1.8, -0.2, -1.0),
            (1.8, 0.2, 1.0),
            (1.99, -0.01, -1.0),
            (1.99, 0.01, 1.0),
        ]
        for alpha, theta, expected_beta in cases:
            beta, scale = from_feller_to_s1(alpha, theta)
            assert np.isfinite(beta) and np.isfinite(scale)
            np.testing.assert_allclose(beta, expected_beta, atol=1e-13)
            np.testing.assert_allclose(
                scale, self.expected_scale(alpha, theta), rtol=1e-13, atol=1e-13
            )

    def test_s1_to_feller_at_beta_boundaries(self):
        cases = [
            (0.01, -1.0, 0.01),
            (0.01, 1.0, -0.01),
            (0.2, -1.0, 0.2),
            (0.2, 1.0, -0.2),
            (0.8, -1.0, 0.8),
            (0.8, 1.0, -0.8),
            (1.2, -1.0, -0.8),
            (1.2, 1.0, 0.8),
            (1.8, -1.0, -0.2),
            (1.8, 1.0, 0.2),
            (1.99, -1.0, -0.01),
            (1.99, 1.0, 0.01),
        ]
        for alpha, beta, expected_theta in cases:
            theta, scale = from_s1_to_feller(alpha, beta)
            assert np.isfinite(theta) and np.isfinite(scale)
            np.testing.assert_allclose(theta, expected_theta, atol=1e-13)
            np.testing.assert_allclose(
                scale, self.expected_scale(alpha, expected_theta),
                rtol=1e-13, atol=1e-13,
            )

    def test_feller_to_s1_round_trip_at_and_near_boundaries(self):
        for alpha in [0.01, 0.2, 0.8, 0.999999, 1.000001, 1.2, 1.8, 1.99]:
            theta_limit = min(alpha, 2.0-alpha)
            theta_values = [
                -theta_limit,
                np.nextafter(-theta_limit, 0.0),
                -theta_limit/2,
                0.0,
                theta_limit/2,
                np.nextafter(theta_limit, 0.0),
                theta_limit,
            ]
            for theta in theta_values:
                beta, scale = from_feller_to_s1(alpha, theta)
                theta_back, scale_back = from_s1_to_feller(alpha, beta)
                assert np.all(np.isfinite([beta, scale, theta_back, scale_back]))
                np.testing.assert_allclose(theta_back, theta, atol=1e-13)
                np.testing.assert_allclose(scale_back, scale, rtol=1e-13, atol=1e-13)

    def test_s1_to_feller_round_trip_at_and_near_boundaries(self):
        beta_values = [
            -1.0,
            np.nextafter(-1.0, 0.0),
            -0.5,
            0.0,
            0.5,
            np.nextafter(1.0, 0.0),
            1.0,
        ]
        for alpha in [0.01, 0.2, 0.8, 0.999999, 1.000001, 1.2, 1.8, 1.99]:
            for beta in beta_values:
                theta, scale = from_s1_to_feller(alpha, beta)
                beta_back, scale_back = from_feller_to_s1(alpha, theta)
                assert np.all(np.isfinite([theta, scale, beta_back, scale_back]))
                np.testing.assert_allclose(beta_back, beta, rtol=2e-10, atol=2e-10)
                np.testing.assert_allclose(scale_back, scale, rtol=2e-10, atol=2e-10)

    def test_requested_interior_example_in_both_directions(self):
        alpha, theta = 1.2, 0.5
        expected_beta = 0.324919696232906
        expected_scale = 0.749153538438341

        beta, scale = from_feller_to_s1(alpha, theta)
        np.testing.assert_allclose([beta, scale], [expected_beta, expected_scale])

        theta_back, scale_back = from_s1_to_feller(alpha, beta)
        np.testing.assert_allclose([theta_back, scale_back], [theta, expected_scale])

    def test_nan_inputs_are_rejected(self):
        with pytest.raises(AssertionError):
            from_feller_to_s1(np.nan, 0.5)
        with pytest.raises(AssertionError):
            from_feller_to_s1(0.8, np.nan)
        with pytest.raises(AssertionError):
            from_s1_to_feller(np.nan, 0.5)
        with pytest.raises(AssertionError):
            from_s1_to_feller(0.8, np.nan)

    def test_theta_outside_feller_diamond_is_rejected(self):
        for alpha in [0.2, 0.8, 1.2, 1.8]:
            theta_limit = min(alpha, 2.0-alpha)
            outside = theta_limit + 1e-10
            with pytest.raises(AssertionError):
                from_feller_to_s1(alpha, outside)
            with pytest.raises(AssertionError):
                from_feller_to_s1(alpha, -outside)
    
class Test_GAS_Equiv_K1:
    alpha = 0.85
    theta = alpha * 0.25
    gas1 = lihn_stable(alpha=alpha, k=1.0, theta=theta)
    gas2 = levy_stable_from_feller(alpha=alpha, theta=theta)
    gas3 = LihnStable(alpha=alpha, k=1.0, theta=theta)

    def test_x1(self):
        x = -0.85
        compare_two_rvs(x, self.gas1, self.gas2, min_p=0.1)

    def test_x2(self):
        x = 0.85
        compare_two_rvs(x, self.gas1, self.gas2, min_p=0.1)

    def test_k1_mellin(self):
        x = 0.45
        eps = self.gas3.eps
        g = self.gas3.g

        def _mellin(s):
            return wright_m_fn_rescaled_mellin_transform(s, g) * fcm_k1_mellin_transform(2.0-s, eps, g)

        p1 = self.gas3.pdf(x)
        p2 = pdf_by_mellin(x, _mellin)
        delta_precise_up_to(p1, p2)
        
    def test_k1_integral(self):
        x = 0.55
        g = self.gas3.g

        def _pdf(s):
            return wright_m_fn_rescaled_by_levy(x*s, g) * self.gas3.fcm.pdf(s) * s  # type: ignore

        p1 = self.gas3.pdf(x)
        p3 = quad(_pdf, a=0, b=np.inf)[0]
        delta_precise_up_to(p1, p3)


class TestGSaS_PdfAtZero:
    alpha = 0.55
    k = 3.0
    p1 = gsas_pdf_at_zero(alpha=alpha, k=k)    # gsas
    p2 = gsas_pdf_at_zero(alpha=alpha, k=1.0)  # sas

    def test_sas(self):
        sas = levy_stable(alpha=self.alpha, beta=0)
        p3 = sas.pdf(0.0)  # type: ignore
        delta_precise_up_to(self.p2, p3)

    def test_gsas(self):
        p3 = gsas(alpha=self.alpha, k=self.k).pdf(0.0)  # type: ignore
        delta_precise_up_to(self.p1, p3)
        p4 = gsas(alpha=self.alpha, k=1.0).pdf(0.0)  # type: ignore
        delta_precise_up_to(self.p2, p4)

    def test_gas(self):
        p3 = lihn_stable(alpha=self.alpha, k=self.k, theta=0.0).pdf(0.0)  # type: ignore
        delta_precise_up_to(self.p1, p3)
        p4 = lihn_stable(alpha=self.alpha, k=1.0, theta=0.0).pdf(0.0)  # type: ignore
        delta_precise_up_to(self.p2, p4)

    def test_std_gsas(self):
        g = gsas(alpha=self.alpha, k=self.k)
        sd = g.moment(2.0)**0.5
        g2 = gsas(alpha=self.alpha, k=self.k, scale=1/sd)
        delta_precise_up_to(g2.pdf(0.0), gsas_std_pdf_at_zero(self.alpha, self.k))  # type: ignore

        sd2 = g2.moment(2.0)**0.5
        delta_precise_up_to(sd2, 1.0)  # g2 is a standardized distribution with sd=1


class TestGSaS_PdfAtZero_V2:
    alpha = 0.6  # 0.33
    k = 4.0  # k >= 5: gas_dist.py:110: IntegrationWarning: The integral is probably divergent, or slowly convergent
    p1 = gsas_pdf_at_zero(alpha=alpha, k=k)    # gsas
    p2 = gsas_pdf_at_zero(alpha=alpha, k=1.0)  # sas

    def test_sas(self):
        sas = levy_stable(alpha=self.alpha, beta=0)
        p3 = sas.pdf(0.0)  # type: ignore
        delta_precise_up_to(self.p2, p3)

    def test_gsas(self):
        p3 = gsas(alpha=self.alpha, k=self.k).pdf(0.0)  # type: ignore
        delta_precise_up_to(self.p1, p3)
        p4 = gsas(alpha=self.alpha, k=1.0).pdf(0.0)  # type: ignore
        delta_precise_up_to(self.p2, p4)

    def test_gas(self):
        p3 = lihn_stable(alpha=self.alpha, k=self.k, theta=0.0).pdf(0.0)  # type: ignore
        delta_precise_up_to(self.p1, p3)
        p4 = lihn_stable(alpha=self.alpha, k=1.0, theta=0.0).pdf(0.0)  # type: ignore
        delta_precise_up_to(self.p2, p4)

    def test_std_gsas(self):
        g = gsas(alpha=self.alpha, k=self.k)
        sd = g.moment(2.0)**0.5
        g2 = gsas(alpha=self.alpha, k=self.k, scale=1/sd)
        delta_precise_up_to(g2.pdf(0.0), gsas_std_pdf_at_zero(self.alpha, self.k))  # type: ignore

        sd2 = g2.moment(2.0)**0.5
        delta_precise_up_to(sd2, 1.0)  # g2 is a standardized distribution with sd=1


class TestGSkew_Alpha_At_1:
    theta = -0.71
    k = 1.5  

    q = np.cos(theta*np.pi/2)
    tau = np.tan(theta*np.pi/2)
    
    def test_simplified_g_skew(self):
        x = 0.45
        s = 0.63

        p1 = g_skew_v2(x, s, alpha=1.0, theta=self.theta, use_short_cut=False)
        y = (self.tau + x/self.q) * s
        p2 = 1/self.q * norm().pdf(y)  # type: ignore
        delta_precise_up_to(p1, p2)
        
        p3 = g_skew_v2(x, s, alpha=1.0, theta=self.theta, use_short_cut=True)
        delta_precise_up_to(p1, p3)


class TestGSkew_Osc:
    def compare_g_skew(self, alpha, theta):
        for x in [-0.5, 0.5]:
            for s in [10.0, 100.0]:
                p1 = g_skew_v2(x, s, alpha=alpha, theta=theta, use_osc=False)
                p2 = g_skew_v2_osc(x, s, alpha=alpha, theta=theta)
                msg = f"alpha={alpha}, theta={theta}, x={x}, s={s}"
                # print(msg)
                delta_precise_up_to(p1, p2, abstol=min(abs(p1),abs(p2))*0.1, msg_prefix=msg)

    def test_g_skew_1(self):
        self.compare_g_skew(alpha=0.85, theta=0.2)

    def test_g_skew_2(self):
        self.compare_g_skew(alpha=0.85, theta=-0.2)

    def test_g_skew_3(self):
        self.compare_g_skew(alpha=1.15, theta=0.25)

    def test_g_skew_4(self):
        self.compare_g_skew(alpha=1.15, theta=-0.25)


# -------------------------------------------------------------------------------------
# SaS ratio tests
class TestSaS_Ratio:
    alpha = 0.55
    x = 0.5
    sym_stable  = levy_stable(alpha=alpha, beta=0, scale=1.0)
    p1 = sym_stable.pdf(x)  # type: ignore
    
    stable_sas_gsc_v1    = gen_stable_count(alpha=alpha,   sigma=1.0,             d=0.0,  p=alpha)  
    stable_sas_gcm_v2    = gen_stable_count(alpha=alpha/2, sigma=1.0/np.sqrt(2),  d=0.0,  p=alpha)  # k=1
    stable_sas_gcm_v3    = lihn_stable.frac_chi_mean(alpha=alpha, k=1.0, theta=0.0)

    def perform_ratio_test(self, unit_fn, gsc):
        ratio_dist_test_suite(self.p1, self.x, unit_fn, gsc)

    def test_sas_gsc_ratio_v1(self):
        self.perform_ratio_test(_cauchy, self.stable_sas_gsc_v1)

    def test_sas_gsc_ratio_v2(self):
        self.perform_ratio_test(_norm, self.stable_sas_gcm_v2)

    def test_sas_gsc_ratio_v3(self):
        self.perform_ratio_test(_norm, self.stable_sas_gcm_v3)

    def test_pdf_at_zero(self):
        p1 = self.sym_stable.pdf(0.0)  # type: ignore
        p2 = gsas_pdf_at_zero(self.alpha, k=1)
        delta_precise_up_to(p1, p2)


# -------------------------------------------------------------------------------------
class Test_GAS_Symmetry:
    alpha = 1.2
    theta = 0.1
    k = 2.5

    def test_g_skew_symmetry(self):
        x = 0.35
        s = 1.45
        p1 = g_skew_v2(-x, s, self.alpha, self.theta)
        p2 = g_skew_v2(x, s, self.alpha, -self.theta)
        delta_precise_up_to(p1, p2)

    def test_gas_symmetry(self):
        x = 0.25
        p1 = lihn_stable(self.alpha, k=self.k, theta=  self.theta).pdf(-x)  # type: ignore
        p2 = lihn_stable(self.alpha, k=self.k, theta= -self.theta).pdf( x)  # type: ignore
        delta_precise_up_to(p1, p2)

    def test_levy_stable_reciprocal(self):
        x = 0.85
        alpha = self.alpha
        beta, scale = from_feller_to_s1(1/alpha, self.theta)
        p1 = levy_stable(1/alpha, beta=beta, scale=scale).pdf(x**(-alpha)) / x**(alpha+1)  # type: ignore

        theta2 = alpha*(self.theta + 1) - 1.0
        beta2, scale2 = from_feller_to_s1(alpha, theta2)
        p2 = levy_stable(alpha, beta=beta2, scale=scale2).pdf(x)  # type: ignore
        delta_precise_up_to(p1, p2)

    def test_afstable_reciprocal(self):
        x = 0.85
        alpha = self.alpha
        k = 1.0  # only 1 works for now
        p1 = lihn_stable(1/alpha, k=k, theta=self.theta).pdf(x**(-alpha)) / x**(alpha+1)  # type: ignore
        theta2 = alpha*(self.theta + 1) - 1.0
        p2 = lihn_stable(alpha, k=k, theta= theta2).pdf(x)  # type: ignore
        delta_precise_up_to(p1, p2)


# -------------------------------------------------------------------------------------
class Test_GAS_Consistency:
    alpha = 1.2
    theta = 0.1
    k = 2.5
    gas1 = lihn_stable(alpha=alpha, k=k, theta=theta)
    gas2 = LihnStable(alpha=alpha, k=k, theta=theta)
    
    x = 0.45 * (-1 if gas2.g < 0.5 else 1)

    def test_mellin_unadjusted(self):
        x = self.x
        p1 = self.gas2.pdf_unadjusted(x)
        p2 = self.gas2.pdf_by_mellin(x)
        delta_precise_up_to(p1, p2)

    def test_pdf(self):
        x = self.x
        p1 = self.gas1.pdf(x)  # type: ignore
        p2 = self.gas2.pdf(x)
        delta_precise_up_to(p1, p2)

    def test_cdf_total_approx(self):
        p1 = self.gas1.cdf(20.0)
        delta_precise_up_to(p1, 1.0, abstol=0.001, reltol=0.001)

    def test_total_density(self):
        p1 = quad_total_density(self.gas1.pdf)  # type: ignore
        delta_precise_up_to(p1, 1.0)

    def test_total_density_unadjusted(self):
        p1 = quad_total_density(self.gas2.pdf_unadjusted, two_sided=False)  # type: ignore
        p2 = self.gas2.g 
        delta_precise_up_to(p1, p2)

    def test_total_density_unadjusted_negative(self):
        g_reflect = self.gas2.new_by_negative_theta()
        p1 = quad_total_density(g_reflect.pdf_unadjusted, two_sided=False)  # type: ignore
        p2 = 1.0 - self.gas2.g 
        delta_precise_up_to(p1, p2)


class Test_GAS_Canonical:
    alpha = 1.2
    theta = 0.1
    k = 2.5
    
    x = 0.45 

    gas1 = LihnStable(alpha=alpha, k=k, theta=theta, slope_sigma_pow_spec='zero')
    assert gas1.slope_sigma_pow == 0.0  # make sure this is truly canonical
    
    def test_pdf_positive(self):
        x = self.x
        p1 = self.gas1.pdf(x)
        p2 = self.gas1.pdf_unadjusted(x) / self.gas1.A_plus()
        delta_precise_up_to(p1, p2)
    
    def test_pdf_negative(self):
        x = -self.x
        p1 = self.gas1.pdf(x)

        c = self.gas1.Psi() / self.gas1.A_plus() / self.gas1.Sigma()
        p2 = c * self.gas1.pdf_unadjusted(x / self.gas1.Sigma())
        delta_precise_up_to(p1, p2)

    def test_total_density_positive(self):
        p1 = quad_total_density(self.gas1.pdf, two_sided=False)  # type: ignore
        p2 = self.gas1.g / self.gas1.A_plus()
        delta_precise_up_to(p1, p2)

    def test_cdf_negative(self):
        p1 = self.gas1.cdf(0.0)
        p2 = (1-self.gas1.g) * self.gas1.Psi() / self.gas1.A_plus() 
        delta_precise_up_to(p1, p2)

    def test_cdf_total_approx(self):
        p1 = self.gas1.cdf(20.0)
        delta_precise_up_to(p1, 1.0, abstol=0.001, reltol=0.001)


# -------------------------------------------------------------------------------------

# GAS ratio tests
class TestGAS_Ratio:
    alpha = 0.55
    theta = 0.25

    beta = 0
    scale = 0

    x = 0.5
    sym_stable  = levy_stable(alpha=alpha, beta=0, scale=1.0)
    p1 = sym_stable.pdf(x)  # type: ignore
    
    stable_sas_gsc_v1    = gen_stable_count(alpha=alpha,   sigma=1.0,             d=0.0,  p=alpha)  
    stable_sas_gcm_v2    = gen_stable_count(alpha=alpha/2, sigma=1.0/np.sqrt(2),  d=0.0,  p=alpha)  # k=1
    stable_sas_gcm_v3    = lihn_stable.frac_chi_mean(alpha=alpha, k=1.0, theta=0.0)

    def perform_ratio_test(self, unit_fn, gsc):
        ratio_dist_test_suite(self.p1, self.x, unit_fn, gsc)

    def test_sas_gsc_ratio_v1(self):
        self.perform_ratio_test(_cauchy, self.stable_sas_gsc_v1)

    def test_sas_gsc_ratio_v2(self):
        self.perform_ratio_test(_norm, self.stable_sas_gcm_v2)

    def test_sas_gsc_ratio_v3(self):
        self.perform_ratio_test(_norm, self.stable_sas_gcm_v3)


# moments
def test_gsas_moments():
    for alpha in [0.55, 0.80, 1.20]:
        for n in [2.0, 4.0]:
            for k in np.arange(n+1, 8.0):
                g = gsas(alpha=alpha, k=k)
                m1 = g.moment(n)
                m2 = gsas_moment(n=n, alpha=alpha, k=k)
                delta_precise_up_to(m1, m2)


# large k formula
