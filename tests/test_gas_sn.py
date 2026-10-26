
import numpy as np
import pytest
from scipy.stats import skewnorm, chi, chi2, f, skew, kurtosis
from scipy.integrate import quad
from scipy.special import owens_t
from scipy.optimize import minimize

from .unit_test_utils import *

from .fcm_dist import frac_chi_mean, fcm_lognormal_scaling_constant
from .gas_sn_dist import gas_sn, SN_Std, SN, ST_Std, ST, GAS_SN_Std, GAS_SN, GSAS, GAS_SN_TAIL,\
    pdf_sqare, gas_sn_mellin_transform, owens_t_mellin_transform
from .gsas_dist import gsas_mixture_pdf


class Test_PDF_Square:
    def test_chi_square(self):
        k = 3.0
        x = 0.5
        p1 = pdf_sqare(lambda x: chi(k).pdf(x), x)  # type: ignore
        p2 = chi2(k).pdf(x)  # type: ignore
        delta_precise_up_to(p1, p2)


class Test_SN:
    beta = -0.8

    sn1 = skewnorm(beta)
    sn2 = SN_Std(beta)
    
    loc = 0.2
    scale = 0.8
    
    sn3 = skewnorm(beta, scale=scale, loc=loc)
    sn4 = SN(beta, scale=scale, loc=loc)
    
    def test_pdf(self):
        for x in [-1.1, 0.0, 1.6]:
            p1 = self.sn1.pdf(x)  # type: ignore
            p2 = self.sn2._pdf(x)
            delta_precise_up_to(p1, p2)

            p3 = self.sn3.pdf(x)  # type: ignore
            p4 = self.sn4.pdf(x)
            delta_precise_up_to(p3, p4)

    def test_cdf(self):
        for x in [-1.1, 0.0, 1.6]:
            p1 = self.sn1.cdf(x)  # type: ignore
            p2 = self.sn2._cdf(x)
            delta_precise_up_to(p1, p2)

            p3 = self.sn3.cdf(x)  # type: ignore
            p4 = self.sn4.cdf(x)
            delta_precise_up_to(p3, p4)

    def test_moments(self):
        for n in [0, 1, 2, 3, 4]:
            p1 = self.sn1.moment(n)
            p2 = self.sn2._moment(n)
            delta_precise_up_to(p1, p2)

            p3 = self.sn2._moment_desciptive(n)
            delta_precise_up_to(p1, p3)

    def test_mean(self):
        p1 = self.sn1.stats('m')  # type: ignore
        p2 = self.sn2._mean()
        delta_precise_up_to(p1, p2)

        p3 = self.sn3.stats('m')  # type: ignore
        p4 = self.sn4.mean() 
        delta_precise_up_to(p3, p4)

    def test_var(self):
        p1 = self.sn1.stats('v')  # type: ignore
        p2 = self.sn2._var()
        delta_precise_up_to(p1, p2)

        p3 = self.sn3.stats('v')  # type: ignore
        p4 = self.sn4.var()
        delta_precise_up_to(p3, p4)

    def test_skew(self):
        p1 = self.sn1.stats('s')  # type: ignore
        p2 = self.sn2._skew()
        delta_precise_up_to(p1, p2)

    def test_kurtosis(self):
        p1 = self.sn1.stats('k')  # type: ignore
        p2 = self.sn2._kurtosis()
        delta_precise_up_to(p1, p2)

    def test_mode(self):
        beta = -0.3  # beta can not be too large, otherwise, p3 will be off too much
        p1 = SN_Std(beta)._mode()
        p2 = SN_Std(-beta)._mode() * -1  # symmetry test
        delta_precise_up_to(p1, p2)

        fn = lambda x: -1 * skewnorm(beta).pdf(x)  # type: ignore
        p3 = minimize(fn, 0).x[0]  
        delta_precise_up_to(p1, p3)
        
        p4 = SN_Std(beta)._find_mode()
        delta_precise_up_to(p1, p4)

    def test_x_square(self):
        x = 0.5
        p1 = self.sn2.rv_square.pdf(x)  # type: ignore
        p2 = chi2(1).pdf(x)  # type: ignore
        delta_precise_up_to(p1, p2)
        
    def test_mgf(self):
        beta = 0.3
        scale = 1.2
        loc = 0.25

        def sn_mgf(t):
            def sn_integrand(x):
                return np.exp(x*t) * skewnorm(beta, scale=scale, loc=loc).pdf(x)  # type: ignore
            return quad(sn_integrand, -np.inf, np.inf)[0]

        t = -0.15
        p1 = sn_mgf(t)
        p2 = SN(beta, scale=scale, loc=loc).mgf(t) 
        delta_precise_up_to(p1, p2)


class Test_SN_RVS:
    beta = 1.5
    sn = SN_Std(beta)
    num_samples = int(1e7)  # each 1e6 takes a second
    z = sn._rvs(num_samples)
    z_sqr = np.array([x**2 for x in z])
               
    def assert_mvsk_for_sn_rvs(self, z, msg):
        tol = [5e-3, 5e-3, 5e-2, 5e-2]  # higher moments are harder to pass!
        assert_mvsk_from_rvs(
            skewnorm(self.beta).stats("mvsk"), 
            z , tol,
            msg_prefix= lambda i: f"SN_RVS {msg}: {i+1}-th mvsk of Z",
            abstol_max=1.0)

    def test_mvsk_from_rvs_v1(self):
        self.assert_mvsk_for_sn_rvs(self.z, msg="v1")

    def test_mvsk_from_rvs_v2(self):
        z2 = self.sn._rvs_v2(self.num_samples)
        self.assert_mvsk_for_sn_rvs(z2, msg="v2")

    def test_mvsk_from_rvs_v3(self):
        z3 = self.sn._rvs_v3(self.num_samples)
        self.assert_mvsk_for_sn_rvs(z3, msg="v3")

    def test_mvsk_from_rvs_v4(self):
        z4 = self.sn._rvs_v4(self.num_samples)
        self.assert_mvsk_for_sn_rvs(z4, msg="v4")
 
    def test_mvsk_from_z_sqr_rvs(self):
        tol = [5e-3, 5e-3, 5e-2, 5e-2]  # Z^2 is harder than Z, and higher moments are harder to pass!
        assert_mvsk_from_rvs(
            self.sn.rv_square.stats("mvsk"),
            self.z_sqr, tol,
            msg_prefix=lambda i: f"SN_RVS {i+1}-th mvsk Z^2",
            abstol_max=1.0)

# --------------------------------------------------------------------
def test_owens_t_mellin_transform():
    for beta in [-0.8, 0.0, 0.6]:
        for s in [0.5, 1.0, 1.5]:
            p1 = owens_t_mellin_transform(s, beta)
            p2 = quad(lambda x: owens_t(x, beta) * x**(s-1), 0, np.inf)[0]
            delta_precise_up_to(p1, p2, msg_prefix=f"beta ={beta}, s={s}")


# --------------------------------------------------------------------
class Test_ST:
    alpha = 1.0
    k = 4.8 
    beta = 0.4

    g = gas_sn(alpha, k, beta)
    st = ST_Std(k, beta)
    g_st = GAS_SN_Std(alpha, k, beta)

    loc = 0.1
    scale = 0.85
    g2 = gas_sn(alpha, k, beta, scale=scale, loc=loc)
    st2 = ST(k, beta, scale=scale, loc=loc)

    def test_pdf(self):
        for x in [-1.1, 0.0, 1.6]:
            p1 = self.g.pdf(x)  # type: ignore
            p2 = self.st._pdf(x)
            delta_precise_up_to(p1, p2)

            p3 = self.g2.pdf(x)  # type: ignore
            p4 = self.st2.pdf(x)
            delta_precise_up_to(p3, p4)

    def test_cdf(self):
        for x in [0.0, 0.2, 1.6]:
            z = (x - self.loc) / self.scale
            p1 = self.g.cdf(z)
            p2 = self.st._cdf(z)
            delta_precise_up_to(p1, p2)

            p3 = self.g2.cdf(x)  # type: ignore
            p4 = self.st2.cdf(x)
            delta_precise_up_to(p3, p4, msg_prefix=f"st2 cdf x={x}")

    def test_skew_cauchy(self):
        for x in [-0.5, 0.0, 1.6]:
            p1 = gas_sn(1.0, 1.0, self.beta).pdf(x)  # type: ignore
            p2 = ST_Std(1.0, self.beta)._pdf(x)
            delta_precise_up_to(p1, p2)

    def test_moments(self):
        for n in [0, 1, 2, 3, 4]:
            p1 = self.st._moment(n)
            p2 = self.g.moment(n)
            delta_precise_up_to(p1, p2)

    def test_mean(self):
        p1 = self.g2.stats('m')  # type: ignore
        p2 = self.st2.mean()
        delta_precise_up_to(p1, p2)

    def test_var(self):
        p1 = self.g2.stats('v')  # type: ignore
        p2 = self.st2.var()
        delta_precise_up_to(p1, p2)
        
    def test_x_square(self):
        x = 0.5
        p1 = self.st.rv_square.pdf(x)  # type: ignore
        p2 = self.g_st.rv_square.pdf(x)
        delta_precise_up_to(p1, p2)


class Test_ST_RVS:
    k = 20.0  # needs to be quite large for smaller Z^2's kurtosis
    beta = 0.5
    st = ST_Std(k, beta)
    num_samples = int(1e7)  # each 1e6 takes a second
    z = st._rvs(num_samples)
    z_sqr = np.array([x**2 for x in z])
               
    def test_mvsk_from_rvs(self):
        tol = [5e-3, 1e-2, 0.1, 0.2]  # higher moments are harder to pass!
        assert_mvsk_from_rvs(
            self.st._stats_mvsk(),
            self.z, tol,
            msg_prefix= lambda i: f"ST_RVS {i+1}-th mvsk Z",
            abstol_max=1.0)

    def test_mvsk_from_z_sqr_rvs(self):
        tol = [5e-3, 0.05, 0.1, 0.2]  # Z^2 is harder than Z, and higher moments are harder to pass!
        # kurtosis can get really bad if k is not high enough
        assert_mvsk_from_rvs(
            self.st.rv_square.stats("mvsk"),
            self.z_sqr, tol, 
            msg_prefix= lambda i: f"ST_RVS {i+1}-th mvsk Z^2",
            abstol_max=4.0)


# --------------------------------------------------------------------
def test_gas_sn_mellin_transform():
    alpha = 1.2
    k = 4.8
    for beta in [-0.8, 0.0, 0.4]:
        for s in [1.0, 2.0, 3.5]:
            p1 = gas_sn_mellin_transform(s, alpha, k, beta, exact_form=False)
            p2 = gas_sn_mellin_transform(s, alpha, k, beta, exact_form=True)
            delta_precise_up_to(p1, p2, msg_prefix=f"beta ={beta}, s={s}")


# --------------------------------------------------------------------
class Test_GAS_SN:
    alpha = 1.2
    k = 4.2
    beta = 0.35

    g = gas_sn(alpha, k, beta)
    g2 = GAS_SN_Std(alpha, k, beta)

    loc = 0.1
    scale = 0.85
    gls = gas_sn(alpha, k, beta, scale=scale, loc=loc)
    gls2 = GAS_SN(alpha, k, beta, scale=scale, loc=loc)

    g0 = GAS_SN_Std(alpha, k, beta=0.0)
    
    def test_pdf(self):
        for x in [-1.1, 0.0, 1.6]:
            p1 = self.g.pdf(x)  # type: ignore
            p2 = self.g2._pdf(x)
            delta_precise_up_to(p1, p2)

            p3 = self.gls.pdf(x)  # type: ignore
            p4 = self.gls2.pdf(x)
            delta_precise_up_to(p3, p4)
            
    def test_pdf_by_mellin(self):
        # this only works for GSaS for now since we don't know how to evaluate T's CDF on the complex plane
        for x in [-0.54, 0.65]:
            p1 = self.g0._pdf(x)
            p2 = self.g0._pdf_by_mellin(x)
            delta_precise_up_to(p1, p2)

    def test_cdf(self):
        for x in [0.0, 0.2, 1.6]:
            p1 = self.g.cdf(x)
            p2 = self.g2._cdf(x)
            delta_precise_up_to(p1, p2)

            p3 = self.gls.cdf(x)  # type: ignore
            p4 = self.gls2.cdf(x)
            delta_precise_up_to(p3, p4)
            
    def test_cdf_by_mellin(self):
        # this only works for GSaS for now since we don't know how to evaluate T's CDF on the complex plane
        for x in [-0.54, 0.65]:
            p1 = self.g0._cdf(x)
            p2 = self.g0._cdf_by_mellin(x)
            delta_precise_up_to(p1, p2)

    def test_moments(self):
        for n in [0, 1, 2, 3, 4]:
            p1 = self.g.moment(n)
            p2 = self.g2._moment(n)
            delta_precise_up_to(p1, p2)

    def test_moments_by_mellin(self):
        for n in [0, 1, 2, 3, 4]:
            p1 = self.g.moment(n)
            p2 = self.g2._moment_by_mellin(n)
            delta_precise_up_to(p1, p2)

    def test_mean(self):
        p1 = self.g.stats('m')  # type: ignore
        p2 = self.g2._mean()
        delta_precise_up_to(p1, p2)

    def test_var(self):
        p1 = self.g.stats('v')  # type: ignore
        p2 = self.g2._var()
        delta_precise_up_to(p1, p2)

    def test_skew_kurtosis(self):
        p1 = self.g.stats('s')
        p2 = self.g2._skew()
        delta_precise_up_to(p1, p2)

        p3 = self.g.stats('k')
        p4 = self.g2._kurtosis()
        delta_precise_up_to(p3, p4)

    def test_skew_formula(self):
        p1 = self.g2._skew()
        p2 = self.g2._skew_formula()
        delta_precise_up_to(p1, p2)
    
    def test_kurtosis_formula(self):
        p1 = self.g2._kurtosis()
        p2 = self.g2._kurtosis_formula()
        delta_precise_up_to(p1, p2)


class Test_GAS_SN_RVS:
    alpha = 1.35
    k = 6.2
    beta = 0.45
    num_samples = int(5e7)  # each 1e6 takes a second

    g = gas_sn(alpha, k, beta)
    g2 = GAS_SN_Std(alpha, k, beta)

    loc = -0.2
    scale = 0.85
    gls = gas_sn(alpha, k, beta, scale=scale, loc=loc)
    gls2 = GAS_SN(alpha, k, beta, scale=scale, loc=loc)

    z = g.rvs(num_samples)
    z2 = g2._rvs(num_samples)

    z_ls = gls.rvs(num_samples)
    z_ls2 = gls2.rvs(num_samples)

    def test_mvsk_from_rvs_g(self):
        tol = [5e-3, 1e-2, 0.1, 0.3]  # higher moments are harder to pass!
        assert_mvsk_from_rvs(
            self.g2._stats_mvsk(),
            self.z, tol,
            msg_prefix= lambda i: f"GAS_SN_RVS {i+1}-th mvsk Z",
            abstol_max=1.0)

    def test_mvsk_from_rvs_g2(self):
        tol = [5e-3, 1e-2, 0.1, 0.3]  # higher moments are harder to pass!
        assert_mvsk_from_rvs(
            self.g2._stats_mvsk(),
            self.z2, tol,
            msg_prefix= lambda i: f"GAS_SN_RVS {i+1}-th mvsk Z2",
            abstol_max=1.0)

    def test_mvsk_from_rvs_gls(self):
        tol = [1e-2, 1e-2, 0.1, 0.3]  # higher moments are harder to pass!
        assert_mvsk_from_rvs(
            self.gls2.stats_mvsk(),
            self.z_ls, tol,
            msg_prefix= lambda i: f"GAS_SN_RVS {i+1}-th mvsk Z_ls",
            abstol_max=1.0)

    def test_mvsk_from_rvs_gls2(self):
        tol = [1e-2, 1e-2, 0.1, 0.3]  # higher moments are harder to pass!
        assert_mvsk_from_rvs(
            self.gls2.stats_mvsk(),
            self.z_ls2, tol,
            msg_prefix= lambda i: f"GAS_SN_RVS {i+1}-th mvsk Z_ls2",
            abstol_max=1.0)


class Test_GAS_SN_TAIL:
    alpha = 1.35
    k = 4.1
    beta = 1.2

    g = GAS_SN_TAIL(alpha, k, beta)
    g2 = gas_sn(alpha, k, beta)

    def test_pdf_approximates_exact_distribution(self):
        """Fail when the tail approximation exceeds 1% error on either side."""
        for x in [-4.0, -3.0, -2.0, 4.0, 3.0, 2.0]:
            exact = self.g2.pdf(x)  # type: ignore
            approx = self.g.pdf(x)
            delta_precise_up_to(exact, approx, reltol=1e-2, msg_prefix=f"x={x}")

        exact_at_zero = self.g2.pdf(0.0)  # type: ignore
        approx_at_zero = self.g.pdf(0.0)
        delta_precise_up_to(exact_at_zero, approx_at_zero, msg_prefix="x=0")


class Test_GAS_SN_TAIL_V2:
    alpha = 1.1
    k = 3.9
    beta = -1.8

    g = GAS_SN_TAIL(alpha, k, beta)
    g2 = gas_sn(alpha, k, beta)

    def test_pdf_approximates_exact_distribution(self):
        """Fail when the tail approximation exceeds 1% error on either side."""
        for x in [-4.0, -3.0, -2.0, 4.0, 3.0, 2.0]:
            exact = self.g2.pdf(x)  # type: ignore
            approx = self.g.pdf(x)
            delta_precise_up_to(exact, approx, reltol=1e-2, msg_prefix=f"x={x}")

        exact_at_zero = self.g2.pdf(0.0)  # type: ignore
        approx_at_zero = self.g.pdf(0.0)
        delta_precise_up_to(exact_at_zero, approx_at_zero, msg_prefix="x=0")


class Test_GAS_SN_Rescaled:
    # GAS_SN(rescaled=True): the scale is that of Z0 / V_hat, V_hat = D^zeta V (fracdist Definition 7.4). Away from the
    # LNN line it must be the book's GAS_SN with the scale multiplied by D^(-zeta), in every quantity.

    @pytest.mark.parametrize("alpha, k", [(0.8884, -3.2558), (0.9, 3.13), (0.0993, 28.154)])
    def test_rescaled_is_the_book_gas_sn_with_scale_times_D(self, alpha, k):
        """Fails when the flag changes anything but the scale: rescaled=True with scale w must be the book's GAS_SN
        with scale w D^(-zeta) -- in self.scale, the log-inversion pdf and the variance (fcm_moment path)."""
        w, beta = 0.7, -0.2
        new = GAS_SN(alpha=alpha, k=k, beta=beta, scale=w, loc=0.1, rescaled=True)
        old = GAS_SN(alpha=alpha, k=k, beta=beta, scale=w * np.exp(-np.sign(k) * fcm_lognormal_scaling_constant(alpha, k)), loc=0.1)
        assert np.isclose(new.scale, old.scale, rtol=1e-14)
        x = 0.1 + np.sqrt(old.var()) * np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        assert np.max(np.abs(new.pdf_by_log_inversion(x) / old.pdf_by_log_inversion(x) - 1)) < 1e-12
        assert np.isclose(new.var_by_log_moment(), old.var(), rtol=1e-10)

    @pytest.mark.parametrize("alpha, k", [(0.8884, -3.2558), (0.5, 6.0), (0.0993, 28.154)])
    def test_moment_by_log_moment_matches_moment(self, alpha, k):
        """Fails when E[X^n] = E[Z0^n] E[V^-n] is assembled wrongly (sign of odd moments, the -n, the SN factor):
        it must equal the fcm_moment path _moment for n = 1..4, and the rescaled one must be D^(-zeta n) times it."""
        g = GAS_SN_Std(alpha=alpha, k=k, beta=-0.2)
        log_D = fcm_lognormal_scaling_constant(alpha, k)
        for n in (1, 2, 3, 4):
            assert np.isclose(g._moment_by_log_moment(n), g._moment(n), rtol=1e-12)
            assert np.isclose(g._moment_by_log_moment(n, rescaled=True),
                              g._moment(n) * np.exp(-np.sign(k) * n * log_D), rtol=1e-12)

    @pytest.mark.parametrize("alpha, k", [(0.8884, -3.2558), (0.5, 6.0)])
    def test_stats_by_log_moment_match_fcm_moment_stats(self, alpha, k):
        """Fails when the log-moment mean / var / skew / kurtosis differ from the fcm_moment path where both work,
        in book units and rescaled (skew and kurtosis are scale-free, so rescaling must not move them)."""
        for rescaled in (False, True):
            g = GAS_SN(alpha=alpha, k=k, beta=-0.2, scale=0.7, loc=0.1, rescaled=rescaled)
            assert np.allclose([g.mean_by_log_moment(), g.var_by_log_moment(), g.skew_by_log_moment(), g.kurtosis_by_log_moment()],
                               [g.mean(), g.var(), g.skew(), g.kurtosis()], rtol=1e-10)


class Test_GSAS_Bear:
    # SPX bear state (S = 1 in bmda global_regime_S_X.csv, 1991-11-15 .. 2026-04-07): (alpha, k) inverted from its
    # (excess kurtosis, SPD), with its sample mean and sd. It sits on the LNN line, where E[log V] = 5157: the book's
    # scale would be e^5157, so GSAS takes the scale of V_hat = D^zeta V (fracdist Definition 7.4), rescaled=True.
    alpha, k = 0.0012859442177304069, 3021.8240998213632
    mean, sd = -0.000796, 0.016723

    def test_pdf_matches_gsas_mixture_pdf(self):
        """Fails when GSAS cannot represent the bear state: the constructor builds frac_chi_mean (sigma underflows),
        the log-inversion grid forms e^{E log V} (overflow), or the scale is off. The reference,
        gsas_dist.gsas_mixture_pdf, works in centred log units and never needs sigma or D."""
        unit = GSAS(alpha=self.alpha, k=self.k, rescaled=True)
        scale_hat = self.sd / np.sqrt(unit.var_by_log_moment())
        assert 0.1 * self.sd < scale_hat < 10 * self.sd        # O(sd) with D, e^5157 without
        rv = GSAS(alpha=self.alpha, k=self.k, scale=scale_hat, loc=self.mean, rescaled=True)
        x = np.array([0.0, 0.005, -0.005, 0.01, -0.01])
        expected = gsas_mixture_pdf(self.alpha, self.k, x, self.mean, self.sd)
        assert np.max(np.abs(rv.pdf_by_log_inversion(x) / expected - 1)) < 1e-6

    def test_reproduces_the_kurtosis_it_was_fitted_to(self):
        """At the bear state the fcm_moment path gives nan (0 * inf); the log moments must give back the sample excess
        kurtosis 5.4009 that (alpha, k) were inverted from, and the zero mean and skew of a symmetric law."""
        g = GSAS(alpha=self.alpha, k=self.k, rescaled=True)
        assert np.isclose(g.kurtosis_by_log_moment(), 5.4009, atol=1e-4)
        assert abs(g.mean_by_log_moment()) < 1e-15 and abs(g.skew_by_log_moment()) < 1e-15
