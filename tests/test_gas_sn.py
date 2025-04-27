
from scipy.stats import skewnorm, chi, chi2, f, skew, kurtosis
from scipy.optimize import minimize

from .unit_test_utils import *

from .fcm_dist import frac_chi_mean
from .gas_sn_dist import gas_sn, SN_Std, SN, ST_Std, ST, GAS_SN_Std, GAS_SN, pdf_sqare, gas_sn_mellin_transform


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
