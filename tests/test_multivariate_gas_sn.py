
import numpy as np
import pandas as pd

from scipy.stats import multivariate_normal, skewnorm, chi2
from scipy.integrate import quad
from numpy.linalg import inv, det  # type: ignore

from .multivariate_sn import *
from .multivariate import Multivariate_GSaS, Multivariate_GSaS_Adp_2D
from .gas_sn_dist import SN, gas_sn, ST
from .unit_test_utils import *


# -----------------------------------------------------------------
# Multivariate_GAS_SN reduce to ST

class Test_Multivariate_GSaS_SN_Reduce_1d:
    alpha = 1.0
    k = 3.0 
    
    scale = 1.5
    loc = 0.2
    
    beta1 = [1.5]
    cov1 = [[scale**2]]
    loc1 = [loc]

    sn1 = Multivariate_SN(cov=cov1, beta=beta1, loc=loc1)
    sn2 = SN(beta=beta1[0], scale=scale, loc=loc)

    st1 = Multivariate_ST(cov=cov1, k=k, beta=beta1, loc=loc1)
    st2 = ST(k=k, beta=beta1[0], scale=scale, loc=loc)

    g1 = Multivariate_GAS_SN(cov=cov1, alpha=alpha, k=k, beta=beta1, loc=loc1)
    g_1d = gas_sn(alpha=alpha, k=k, beta=beta1[0], scale=scale, loc=loc)

    # try something nether ST nor SN
    alpha2 = 1.2
    g21 = gas_sn(alpha=alpha2, k=k, beta=beta1[0], scale=scale, loc=loc)
    g22 = Multivariate_GAS_SN(cov=cov1, alpha=alpha2, k=k, beta=beta1, loc=loc1)


    def test_pdf_1d_st(self):
        # should reduce to ST in 1d case
        for x in [-0.1, 0.0, 0.1, 0.2]:
            x1 = np.array([x])
            p1 = self.st1.pdf(x1)
            p2 = self.g1.pdf(x1)
            delta_precise_up_to(p1, p2)

            p3 = self.g_1d.pdf(x)  # type: ignore
            delta_precise_up_to(p1, p3)

            p4 = self.st2.pdf(x)
            delta_precise_up_to(p1, p4)

    def test_pdf_1d_sn(self):
        # should reduce to SN in 1d case
        for x in [-0.1, 0.0, 0.1, 0.2]:
            x1 = np.array([x])
            p1 = self.sn1.pdf(x1)
            p2 = skewnorm.pdf(x, self.beta1[0], scale=self.scale, loc=self.loc)
            delta_precise_up_to(p1, p2)
            
            p3 = self.sn2.pdf(x)
            delta_precise_up_to(p1, p3)

    def test_pdf_1d_gas_sn(self):
        # should reduce to SN in 1d case
        for x in [-0.1, 0.0, 0.1, 0.2]:
            x1 = np.array([x])
            p1 = self.g21.pdf(x)  # type: ignore
            p2 = self.g22.pdf(x1)
            delta_precise_up_to(p1, p2)


class Test_Multivariate_GSaS_SN_Reduce_2d:
    alpha = 1.0
    k = 3.0 
    cov = [[2.1, -0.4], [-0.4, 1.5]]
    beta0 = [0.0, 0.0]

    st0 = Multivariate_ST(cov=cov, k=k, beta=beta0)
    g0 = Multivariate_GAS_SN(cov=cov, alpha=alpha, k=k, beta=beta0)
    gsas0 = Multivariate_GSaS(cov=cov, alpha=alpha, k=k)
    
    beta = [1.0, 2.0]
    n = len(beta)
    st = Multivariate_ST(cov=cov, k=k, beta=beta)
    g = Multivariate_GAS_SN(cov=cov, alpha=alpha, k=k, beta=beta)

    # Std version
    st2 = Multivariate_ST(cov=st.corr, k=k, beta=beta)
    g2 = Multivariate_GAS_SN_Std(corr=st.corr, alpha=alpha, k=k, beta=beta)

    def test_pdf_2d_st_beta0(self):
        n = len(self.beta0)
        for y in [-0.3, 0.0, 0.1, 0.2]:
            x1 = [0.1 + y + x*0.1 for x in range(int(n))]
            x1 = np.array(x1)

            p1 = self.st0.pdf(x1)
            p2 = self.g0.pdf(x1)
            delta_precise_up_to(p1, p2)
            
            p3 = self.gsas0.pdf(x1)
            delta_precise_up_to(p1, p3)

    def test_pdf_2d_beta0(self):
        for alpha in [0.85, 1.0, 1.5, 2.0]:
            for k in [1.0, 2.0, 3.0]:
                g = Multivariate_GAS_SN(cov=self.cov, alpha=alpha, k=k, beta=self.beta0)
                gsas = Multivariate_GSaS(cov=self.cov, alpha=alpha, k=k)
                for x in [[0.1, 0.2], [0.3, 0.4]]:
                    p1 = g.pdf(x)
                    p2 = gsas.pdf(x)
                    delta_precise_up_to(p1, p2)

    def test_pdf_2d_st(self):
        for y in [-0.3, 0.0, 0.1, 0.2]:
            x1 = [0.1 + y + x*0.1 for x in range(int(self.n))]
            x1 = np.array(x1)

            p1 = self.st.pdf(x1)
            p2 = self.g.pdf(x1)
            delta_precise_up_to(p1, p2)

            p3 = self.st.pdf_via_gsas(x1)
            delta_precise_up_to(p1, p3)

    def test_pdf_2d_st_std(self):
        for y in [-0.3, 0.0, 0.1, 0.2]:
            x1 = [0.1 + y + x*0.1 for x in range(int(self.n))]
            x1 = np.array(x1)

            p1 = self.st2.pdf(x1)
            p2 = self.g2._pdf(x1)
            delta_precise_up_to(p1, p2)


# -----------------------------------------------------------------
# Multivariate_GAS_SN direct test for alpha != 1
class Test_Multivariate_GAS_SN_2d:
    alpha = 1.2
    k = 4.5
    cov = [[2.1, -0.4], [-0.4, 1.5]]
    beta = [-0.6, -0.3]
    loc = [0.1, 0.2]
    
    g = Multivariate_GAS_SN(cov=cov, alpha=alpha, k=k, beta=beta, loc=loc)
    g2 = Multivariate_GAS_SN_Std(corr=g.corr, alpha=alpha, k=k, beta=beta)

    # Canonical
    gcan = Cannonical_GAS_SN(2, alpha, k, beta_star=g.beta_star())
    # Canonical tansform
    g_ct = Cannonical_GAS_SN_Transform(g)
    
    # rvs
    size = 800000
    X1 = g._rvs(size)
    X2 = g.rvs(size)
    X1_v2 = g._rvs_v2(size)  # this is slow !!! takes one minute to do 100k
    q_mean = g.get_squared_rv().mean()  # type: ignore
    
    def test_pdf(self):
        for y in [-0.3, 0.0, 0.1, 0.2]:
            x1 = [0.05 + y + x*0.1 for x in range(int(self.g.n))]
            x1 = np.array(x1)
            z1 = self.g.w_inv @ (x1 - self.g.loc)

            p1 = self.g.pdf(x1)
            p2 = self.g2._pdf(z1) * det(self.g.w_inv)
            delta_precise_up_to(p1, p2)

    def test_pdf_canonical(self):
        x = [-0.15, -0.25]
        p1 = self.gcan.pdf(x)
        p2 = self.gcan.pdf_prod(x)
        delta_precise_up_to(p1, p2)

    def test_pdf_round_trip_2(self):
        x2 = np.array([0.1, -0.2])

        p1 = self.g.pdf(x2)
        p2 = self.g_ct.pdf_by_can(x2)
        delta_precise_up_to(p1, p2)
        
    def test_quadratic_form(self):
        Q1 = quadratic_form(self.X1, self.g.corr)
        Q2 = self.g.quadratic_form(self.X2) # location scale adj
        assert allclose(self.q_mean, np.mean(Q1), rtol=0.02)
        assert allclose(self.q_mean, np.mean(Q2), rtol=0.02)
        
    def test_quadratic_form_v2(self):
        Q = quadratic_form(self.X1_v2, self.g.corr)
        assert allclose(self.q_mean, np.mean(Q), rtol=0.02)

    def test_rvs_mean_corr(self):
        for i in [0, 1]:
            p1 = self.g._mean()[i]
            if abs(p1) < 0.2: continue
            p2 = np.mean(self.X1[:, i])
            p3 = np.mean(self.X1_v2[:, i])
            print(f"rvs mean (corr): {p1}, {p2}, {p3}")
            delta_precise_up_to(p1, p2, abstol=0.01, reltol=0.01)
            delta_precise_up_to(p1, p3, abstol=0.01, reltol=0.01)

    def test_rvs_mean_cov(self):
        for i in [0, 1]:
            p1 = self.g.mean()[i]
            if abs(p1) < 0.2: continue
            p2 = np.mean(self.X2[:, i])
            print(f"rvs mean (cov): {p1}, {p2}")
            delta_precise_up_to(p1, p2, abstol=0.01, reltol=0.02)


class Test_Multivariate_GAS_SN_3d:
    alpha = 1.25
    k = 4.3
    cov = np.array([[2.1, 0.4, 0.2], [0.4, 1.5, 0.3], [0.2, 0.3, 1.3]])
    beta = [0.7, -0.4, 0.5]
    loc = [0.1, 0.2, -0.2]
    
    g = Multivariate_GAS_SN(cov=cov, alpha=alpha, k=k, beta=beta, loc=loc)
    g2 = Multivariate_GAS_SN_Std(corr=g.corr, alpha=alpha, k=k, beta=beta)

    # Canonical
    gcan = Cannonical_GAS_SN(3, alpha, k, beta_star=g.beta_star())
    # Canonical tansform
    g_ct = Cannonical_GAS_SN_Transform(g)

    # rvs
    size = 800000
    X1 = g._rvs(size)
    X2 = g.rvs(size)
    q_mean = g.get_squared_rv().mean()  # type: ignore

    def test_pdf(self):
        for y in [-0.3, 0.0, 0.1, 0.2]:
            x1 = [0.05 + y + x*0.1 for x in range(int(self.g.n))]
            x1 = np.array(x1)
            z1 = self.g.w_inv @ (x1 - self.g.loc)

            p1 = self.g.pdf(x1)
            p2 = self.g2._pdf(z1) * det(self.g.w_inv)
            delta_precise_up_to(p1, p2)

    def test_pdf_canonical(self):
        x = np.array([0.5, 0.3, -0.35])
        p1 = self.gcan.pdf(x)
        p2 = self.gcan.pdf_prod(x)
        delta_precise_up_to(p1, p2)

        # pdf_prod is singular at x=0
        x0 = np.array(x) * 0
        assert (x0 @ x0) == 0.0
        p3 = self.gcan.pdf(x0)
        p4 = self.gcan.pdf_prod(x0)
        delta_precise_up_to(p3, p4)

    def test_pdf_round_trip_3(self):
        x = np.array([0.1, -0.2, 0.5])

        p1 = self.g.pdf(x)
        p2 = self.g_ct.pdf_by_can(x)
        delta_precise_up_to(p1, p2)

    def test_quadratic_form(self):
        Q1 = quadratic_form(self.X1, self.g.corr)
        Q2 = self.g.quadratic_form(self.X2) # location scale adj
        assert allclose(self.q_mean, np.mean(Q1), rtol=0.02)
        assert allclose(self.q_mean, np.mean(Q2), rtol=0.02)

    def test_rvs_mean_corr(self):
        for i in [0, 1, 2]:
            p1 = self.g._mean()[i]
            if abs(p1) < 0.2: continue
            p2 = np.mean(self.X1[:, i])
            print(f"rvs mean (corr): {p1}, {p2}")
            delta_precise_up_to(p1, p2, abstol=0.01, reltol=0.01)

    def test_rvs_mean_cov(self):
        for i in [0, 1, 2]:
            p1 = self.g.mean()[i]
            if abs(p1) < 0.2: continue
            p2 = np.mean(self.X2[:, i])
            print(f"rvs mean (cov): {p1}, {p2}")
            delta_precise_up_to(p1, p2, abstol=0.01, reltol=0.01)


# ---------------------------------------------------------------
class Test_GAS_SN_Adp_2D:
    n = 2.0
    cov = np.array([[2.1, 0.4], [0.4, 1.5]])
    alpha = [0.8, 1.3]
    k = [4.5, 5.5]
    beta = [0.5, -0.35]

    loc = [0.1, 0.2]
    beta0 = [0.0, 0.0]

    g = Multivariate_GAS_SN_Adp_2D(cov, alpha, k, beta=beta, loc=loc)

    g_cov_b0 = Multivariate_GAS_SN_Adp_2D(cov, alpha, k, beta=beta0)
    gsas_cov = Multivariate_GSaS_Adp_2D(cov, alpha, k)
    
    corr = g_cov_b0.corr
    g_corr = Multivariate_GAS_SN_Std_Adp_2D(corr, alpha, k, beta)
    g_corr_b0 = Multivariate_GAS_SN_Std_Adp_2D(corr, alpha, k, beta=beta0)

    gsas_corr = Multivariate_GSaS_Adp_2D(corr, alpha, k)
    
    # Canonical
    gcan = Cannonical_GAS_SN_Adp_2D(alpha, k, beta_star=g.beta_star())
    # Canonical tansform
    g_ct = Cannonical_GAS_SN_Adp_2D_Transform(g)

    def test_pdf_equal_gsas(self):
        x = [0.1, 0.2]
        p1 = self.gsas_cov.pdf(x)
        p2 = self.g_cov_b0.pdf(x)
        delta_precise_up_to(p1, p2)

        p3 = self.gsas_corr.pdf(x)
        p4 = self.g_corr_b0._pdf(x)
        delta_precise_up_to(p3, p4)
        
    def test_pdf_std(self):
        x = [-0.15, 0.25]
        z = self.g.w_inv @ (x - self.g.loc)
        p1 = self.g.pdf(x)
        p2 = self.g_corr._pdf(z) * det(self.g.w_inv)
        delta_precise_up_to(p1, p2)

    def test_pdf_canonical(self):
        x = [-0.15, -0.25]
        p1 = self.gcan.pdf(x)
        p2 = self.gcan.pdf_prod(x)
        delta_precise_up_to(p1, p2)

    def test_pdf_round_trip_2(self):
        x2 = np.array([0.1, -0.2])

        p1 = self.g.pdf(x2)
        p2 = self.g_ct.pdf_by_can(x2)
        # delta_precise_up_to(p1, p2)  # TODO don't think will work
