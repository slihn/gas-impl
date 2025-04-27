import numpy as np
import pandas as pd

from scipy.stats import multivariate_normal, multivariate_t, skewnorm, chi2, f
from scipy.integrate import quad
from numpy.linalg import inv, det  # type: ignore

from .multivariate_sn import *
from .multivariate import Multivariate_GSaS, Multivariate_T
from .gas_sn_dist import SN, ST
from .unit_test_utils import *


# -----------------------------------------------------------------
class Test_Multivariate_SN_2d:
    n = 2.0
    cov = [[2.1, 0.4], [0.4, 1.5]]
    sn0 = Multivariate_SN(cov=cov, beta=[0.0, 0.0])
    m_norm = multivariate_normal(cov=cov)  # type: ignore

    beta = [ 0.70, -0.60 ]
    loc = [0.11, -0.12]
    
    sn1 = Multivariate_SN(cov=cov, beta=beta, loc=loc)
    sn2 = Multivariate_SN_Std(corr=sn1.corr, beta=beta)
    
    x = [0.2, 0.3]

    def test_pdf_beta0(self):
        p1 = self.sn0.pdf(self.x)
        p2 = self.m_norm.pdf(self.x)
        delta_precise_up_to(p1, p2)

    def test_unity(self):
        p1 = 1.0         
        p2 = self.sn1.moment_by_2d_int(0, 0)
        delta_precise_up_to(p1, p2)
        
        p3 = self.sn2._moment_by_2d_int(0, 0)
        delta_precise_up_to(p1, p3)

    def test_mean(self):
        m1 = self.sn1.mean()
        m2 = [self.sn1.moment_by_2d_int(1, 0), self.sn1.moment_by_2d_int(0, 1)]
        delta_precise_up_to(m1[0], m2[0], msg_prefix="mean[0]")
        delta_precise_up_to(m1[1], m2[1], msg_prefix="mean[1]")

    def test_var(self):
        var1 = self.sn1.var()
        
        m1 = self.sn1.mean()
        mm = np.outer(m1, m1)
        v2 = [
            self.sn1.moment_by_2d_int(2, 0) - mm[0,0], 
            self.sn1.moment_by_2d_int(0, 2) - mm[1,1], 
            self.sn1.moment_by_2d_int(1, 1) - mm[0,1],
        ]

        delta_precise_up_to(var1[0,0], v2[0], msg_prefix="var[0,0]")
        delta_precise_up_to(var1[1,1], v2[1], msg_prefix="var[1,1]")
        delta_precise_up_to(var1[0,1], v2[2], msg_prefix="var[0,1]")


class Test_Multivariate_SN_3d:
    n = 3.0
    cov = np.array([[2.1, 0.4, 0.2], [0.4, 1.5, 0.3], [0.2, 0.3, 1.4]])
    sn0 = Multivariate_SN(cov=cov, beta=[0.0, 0.0, 0.0])
    m_norm = multivariate_normal(cov=cov)  # type: ignore

    x = [0.2, 0.3, 0.45]

    def test_pdf_beta0(self):
        p1 = self.sn0.pdf(self.x)
        p2 = self.m_norm.pdf(self.x)
        delta_precise_up_to(p1, p2)


class Test_Multivariate_SN_RVS_2d:
    rho = -0.5
    corr = np.array([[1.0, rho], [rho, 1.0]])
    beta = np.array([-1.85, 2.15])  # large enough to have skewnes and kurtosis
    cov = [[2.1, 0.4], [0.4, 1.5]]
    loc = [0.1, 0.2]
    
    gs = Multivariate_SN_Std(corr, beta=beta)
    Z = gs._rvs(5 * 1000 * 1000)

    sn1 = Multivariate_SN(cov=cov, beta=beta, loc=loc)
    Z1  = sn1.rvs(1000 * 1000)
    d = 2.0

    def stats_allclose(self, n):
        stats = calc_mvsk_stats(self.Z[:, n])
        assert abs(stats[2]) > 0.1  # large skew
        assert abs(stats[3]) > 0.1  # large kurtosis

        assert np.allclose(  # type: ignore
            stats, 
            self.gs._marginal_1d_rv(n)._stats_mvsk(), 
            atol=1e-2)

    def test_stats_0(self):
        self.stats_allclose(0)
    
    def test_stats_1(self):
        self.stats_allclose(1)

    def test_squared_stats(self):
        inv_corr = inv(self.gs.corr)
        Z_sqr = np.array([ z1.T @ inv_corr @ z1 for z1 in self.Z ])  # type: ignore
        np.allclose( # type: ignore
            calc_mvsk_stats(Z_sqr), 
            chi2(df=2).stats('mvsk'), 
            atol=5e-2)

    def test_quadratic_form(self):
        Q = quadratic_form(self.Z1, self.sn1.cov)
        np.allclose( # type: ignore
            np.mean(Q), self.sn1.get_squared_rv().mean(),  # type: ignore
            atol=1e-2)  # this is just 1.0 when d = 2


# -----------------------------------------------------------------
class Test_Multivariate_SN_Caononical:
    cov2 = [[2.1, -0.4], [-0.4, 1.5]]
    beta2 = [ 0.70, -0.60 ]
    sn2 = Multivariate_SN(cov=cov2, beta=beta2)
    H2 = sn2.canonicalizer()

    cov3 = [[2.1, 0.4, 0.2], [0.4, 1.5, 0.3], [0.2, 0.3, 1.4]]
    beta3 = np.array([ 0.30, -0.60, 0.40 ])
    sn3 = Multivariate_SN(cov=cov3, beta=beta3)
    H3 = sn3.canonicalizer()

    can2 = Cannonical_SN_Transform(sn2)
    can3 = Cannonical_SN_Transform(sn3)

    sn3_neg = Multivariate_SN(cov=cov3, beta=beta3 * -1)
    can3_neg = Cannonical_SN_Transform(sn3_neg)
    
    def test_is_canonical(self):
        assert self.can2.is_canonical()
        assert self.can3.is_canonical()
    
    def test_beta_star(self):
        p1 = self.sn2.beta_star()
        p2 = self.can2.beta_star()
        delta_precise_up_to(p1, p2)
        
        p3 = self.sn3.beta_star()
        p4 = self.can3.beta_star()
        delta_precise_up_to(p3, p4)
        
    def test_delta_star(self):
        p1 = self.sn2.delta_star()
        p2 = self.can2.delta_star()
        delta_precise_up_to(p1, p2)
        
        p3 = self.sn3.delta_star()
        p4 = self.can3.delta_star()
        delta_precise_up_to(p3, p4)
        
    def test_equals(self):
        assert self.can2.equals(Cannonical_SN(2, self.sn2.beta_star()))
        assert self.can3.equals(Cannonical_SN(3, self.sn3.beta_star()))
    
    def test_pdf_round_trip_2(self):
        x2 = np.array([0.1, -0.2])

        p1 = self.sn2.pdf(x2)
        p2 = self.can2.pdf_by_can(x2)
        delta_precise_up_to(p1, p2)

    def test_pdf_round_trip_3(self):
        x3 = np.array([0.1, 0.1, -0.1])

        p1 = self.sn3.pdf(x3)
        p2 = self.can3.pdf_by_can(x3) 
        delta_precise_up_to(p1, p2) 

        p1 = self.sn3_neg.pdf(x3)
        p2 = self.can3_neg.pdf_by_can(x3) 
        delta_precise_up_to(p1, p2) 

    def test_pdf_prod(self):
        x2 = np.array([0.1, -0.2])        
        x3 = np.array([0.2, -0.1, -0.1])

        p2 = self.can2.pdf(x2)        
        p2a = self.can2.pdf_prod(x2)
        delta_precise_up_to(p2, p2a)

        p3 = self.can3.pdf(x3)
        p3a = self.can3.pdf_prod(x3)
        delta_precise_up_to(p3, p3a)

    def test_pdf_2parts(self):
        x2 = np.array([0.1, -0.2])  
        x3 = np.array([0.2, -0.1, -0.1])

        p2 = self.can2.pdf(x2)        
        p2a = self.can2.pdf_2parts(x2)
        delta_precise_up_to(p2, p2a, msg_prefix="2d")

        p3 = self.can3.pdf(x3)
        p3a = self.can3.pdf_2parts(x3)
        delta_precise_up_to(p3, p3a, msg_prefix="3d")

    def test_multivariate_normal(self):
        x2 = np.array([0.1, -0.2])        
        x3 = np.array([0.2, -0.1, -0.1])
 
        p2 = Cannonical_SN(2, 0.0).pdf(x2)        
        p2a = multivariate_normal(cov=np.eye(2)).pdf(x2)  # type: ignore
        delta_precise_up_to(p2, p2a)

        p3 = Cannonical_SN(3, 0.0).pdf(x3)
        p3a = multivariate_normal(cov=np.eye(3)).pdf(x3)  # type: ignore
        delta_precise_up_to(p3, p3a)

    def test_mode(self):
        self.sn2.find_mode(check=True)
        self.sn3.find_mode(check=True)
    
    def test_marginal_2d(self):
        x = 0.25
        for i in [0, 1]:
            p1 = self.can2.marginal_1d_pdf_by_int(x, i)
            p2 = self.can2.marginal_1d_pdf(x, i)
            delta_precise_up_to(p1, p2)
            
            p2a = SN(self.can2.beta_star()).pdf(x) if i == 0 else skewnorm.pdf(x, 0.0)
            delta_precise_up_to(p2, p2a)
            
            p3 = self.sn2.marginal_1d_pdf_by_int(x, i)
            p4 = self.sn2.marginal_1d_pdf(x, i)
            delta_precise_up_to(p3, p4)

    def test_marginal_3d(self):
        x = 0.15
        for i in [0, 1, 2]:
            p1 = self.can3.marginal_1d_pdf_by_int(x, i)
            p2 = self.can3.marginal_1d_pdf(x, i)
            delta_precise_up_to(p1, p2)
            
            p2a = SN(self.can3.beta_star()).pdf(x) if i == 0 else skewnorm.pdf(x, 0.0)
            delta_precise_up_to(p2, p2a)
            
            p3 = self.sn3.marginal_1d_pdf_by_int(x, i)
            p4 = self.sn3.marginal_1d_pdf(x, i)
            delta_precise_up_to(p3, p4)


# -----------------------------------------------------------------
class Test_Multivariate_T_2d:
    n = 2.0
    cov = [[2.1, 0.4], [0.4, 1.5]]
    alpha = 1.0 
    k = 3.0
    
    mt = multivariate_t(shape=cov, df=k)  # type: ignore
    azz_t = Multivariate_T(cov=cov, k=k)
    g = Multivariate_GSaS(cov=cov, alpha=1.0, k=k)
    g2 = Multivariate_GAS_SN(cov=cov, alpha=1.0, k=k, beta=[0.0, 0.0])

    x = [0.2, 0.3]

    def test_pdf(self):

        p1 = self.mt.pdf(self.x)
        p2 = self.g.pdf(self.x)
        p3 = self.g2.pdf(self.x)
        p4 = self.azz_t.pdf(self.x)

        delta_precise_up_to(p1, p2)
        delta_precise_up_to(p1, p3)
        delta_precise_up_to(p1, p4)

    def test_unity(self):
        p1 = 1.0         
        p2 = self.azz_t.moment_by_2d_int(0, 0)
        delta_precise_up_to(p1, p2)


class Test_Multivariate_T_3d:
    n = 3.0
    cov = np.array([[2.1, 0.4, 0.2], [0.4, 1.5, 0.3], [0.2, 0.3, 1.3]])
    alpha = 1.0 
    k = 4.0
    
    mt = multivariate_t(shape=cov, df=k)  # type: ignore
    azz_t = Multivariate_T(cov=cov, k=k)
    g = Multivariate_GSaS(cov=cov, alpha=1.0, k=k)
    
    x = [0.2, 0.3, 0.4]

    def test_pdf(self):

        p1 = self.mt.pdf(self.x)
        p2 = self.g.pdf(self.x)
        p3 = self.azz_t.pdf(self.x)

        delta_precise_up_to(p1, p2)
        delta_precise_up_to(p1, p3)


class Test_Multivariate_ST_RVS_2d:
    cov = [[2.1, 0.4], [0.4, 1.5]]
    beta = np.array([-1.85, 2.15])  # large enough to have means
    k = 6.0  # large enough to converge
    loc = [0.1, 0.2]

    st = Multivariate_ST(cov=cov, k=k, beta=beta, loc=loc)
    Z = st.rvs(5 * 1000 * 1000)
    z0 = Z[:, 0]
    z1 = Z[:, 1]

    def test_mean(self):
        assert allclose([self.z0.mean(), self.z1.mean()], self.st.mean(), atol=1e-2, rtol=1e-2)
    
    def test_var(self):
        assert allclose(np.cov(self.z0, self.z1), self.st.var(), atol=1e-2, rtol=1e-2)  # type: ignore

    def test_squared_stats(self):
        inv_cov = inv(self.st.cov)
        Y = np.array([ z - self.st.loc for z in self.Z ])
        Z_sqr = np.array([ y.T @ inv_cov @ y for y in Y ]) / self.st.n  # type: ignore
        np.allclose( # type: ignore
            calc_mvsk_stats(Z_sqr)[:2], 
            f(self.st.n, self.st.k).stats('mv'),   # only mean and var exist, no skew or kurt
            atol=5e-2)

    def test_quadratic_form(self):
        Q = quadratic_form(self.Z, self.st.cov)
        np.allclose( # type: ignore
            np.mean(Q), self.st.get_squared_rv().mean(),  # type: ignore
            atol=1e-2)  # this is just 1.0 when d = 2
