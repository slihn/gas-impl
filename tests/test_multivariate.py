
# test gsas

import numpy as np
import pandas as pd
import mpmath as mp
from scipy.special import gamma
from scipy.integrate import quad
from scipy import stats
from scipy.stats import multivariate_normal, multivariate_t, t


from gas_impl.multivariate import Multivariate_GSaS, Multivariate_GSaS_2D
from gas_impl.gas_dist import gsas
from gas_impl.fcm_dist import fcm_moment
from gas_impl.unit_test_utils import *

def _x0(n): return [0.0 for x in range(int(n))]
def _cov_id(n): return np.identity(int(n))


class Test_MGSaS_T_Equiv_2d:
    cov = np.array([[2.1, 0.4], [0.4, 1.5]])
    n = 2.0
    k = 3.0
    x0 = _x0(n)
    x1 = [0.2, 0.3]

    rvt = multivariate_t(x0, cov, df=k)
    mgsas = Multivariate_GSaS(cov=cov, alpha=1.0, k=k)

    def test_pdf_equiv_at_x0(self):
        p1 = self.rvt.pdf(self.x0)
        p2 = self.mgsas.pdf(self.x0)
        delta_precise_up_to(p1, p2)

    def test_pdf_equiv_at_x1(self):
        p1 = self.rvt.pdf(self.x1)
        p2 = self.mgsas.pdf(self.x1)
        delta_precise_up_to(p1, p2)

    def test_x_list(self):
        xs = [self.x1, self.x1, np.array(self.x1)]
        p1 = self.mgsas.pdf(xs)
        p2 = self.mgsas.pdf(np.array(xs))
        delta_precise_up_to(sum(p1), sum(p2))


class Test_MGSaS_T_Equiv_3d:
    cov = np.array([[2.1, 0.4, 0.2], [0.4, 1.5, 0.3], [0.2, 0.3, 1.3]])
    n = 3.0
    k = 4.0
    x0 = _x0(n)
    x1 = [0.2, 0.3, 0.4]

    rvt = multivariate_t(x0, cov, df=int(k))
    mgsas = Multivariate_GSaS(cov=cov, alpha=1.0, k=k)

    def test_pdf_equiv_at_x0(self):
        p1 = self.rvt.pdf(self.x0)
        p2 = self.mgsas.pdf(self.x0)
        delta_precise_up_to(p1, p2)

    def test_pdf_equiv_at_x1(self):
        p1 = self.rvt.pdf(self.x1)
        p2 = self.mgsas.pdf(self.x1)
        delta_precise_up_to(p1, p2)


class Test_MGSaS_2d:
    cov = np.array([[2.1, 0.4], [0.4, 1.5]])
    alpha = 0.9
    k = 3.5
    mgsas = Multivariate_GSaS(cov=cov, alpha=alpha, k=k)

    def test_peak_pdf(self):
        p1 = self.mgsas.pdf(self.mgsas.x0)
        p2 = self.mgsas.pdf_at_zero()
        delta_precise_up_to(p1, p2)


class Test_MGSaS_3d:
    cov = np.array([[2.1, 0.4, 0.2], [0.4, 1.5, 0.3], [0.2, 0.3, 1.3]])
    alpha = 0.9
    k = 4.0
    mgsas = Multivariate_GSaS(cov=cov, alpha=alpha, k=k)

    def test_peak_pdf(self):
        p1 = self.mgsas.pdf(self.mgsas.x0)
        p2 = self.mgsas.pdf_at_zero()
        delta_precise_up_to(p1, p2)

    def test_variance(self):
        var = self.mgsas.variance
        p1 = self.mgsas.marginal_1d_rv(0).moment(2)
        p2 = var[0,0]
        delta_precise_up_to(p1, p2)

        p3 = self.mgsas.marginal_1d_rv(1).moment(2)
        p4 = var[1,1]
        delta_precise_up_to(p3, p4)

        
class Test_MGSaS_MVNormal:
    def test_2d_pdf_equiv_at_x0(self):
        n = 2.0
        cov = _cov_id(n)
        x0 = _x0(n)

        alpha = 1.9
        var = alpha**(2/alpha)
        mgsas = Multivariate_GSaS(cov=cov/var, alpha=alpha, k=10.0)
        p1 = mgsas.pdf(x0) 
        p2 = multivariate_normal(x0, cov).pdf(x0)
        delta_precise_up_to(p1, p2)

    def test_3d_pdf_equiv_at_x0(self):
        n = 3.0
        cov = _cov_id(n)
        x0 = _x0(n)

        alpha = 1.9
        var = alpha**(2/alpha)
        mgsas = Multivariate_GSaS(cov=cov/var, alpha=alpha, k=10.0)
        p1 = mgsas.pdf(x0) 
        p2 = multivariate_normal(x0, cov).pdf(x0)
        delta_precise_up_to(p1, p2, abstol=0.001, reltol=0.01)


# ---------------------------------------------------------------
class Test_MGSaS_2D:
    n = 2.0
    cov = np.array([[2.1, 0.4], [0.4, 1.5]])
    mgsas = Multivariate_GSaS_2D(cov=cov, alpha=[1.1, 0.9], k=[3.0, 4.0])
    
    mm1 = mgsas.fcm_moments(-1.0)
    mm2 = mgsas.fcm_moments(-2.0)

    def test_peak_pdf(self):
        p1 = self.mgsas.pdf(self.mgsas.x0)
        p2 = self.mgsas.pdf_at_zero()
        delta_precise_up_to(p1, p2)

    def test_variance(self):
        var = self.mgsas.variance
        p1 = self.mm2[0] * self.mgsas.cov[0,0]
        p2 = var[0,0]
        delta_precise_up_to(p1, p2)

        p3 = self.mm2[1] * self.mgsas.cov[1,1]
        p4 = var[1,1]
        delta_precise_up_to(p3, p4)

    def test_variance_01(self):
        var = self.mgsas.variance
        p5 = self.mm1[0] * self.mm1[1] * self.mgsas.cov[0,1]
        p6 = var[0,1]
        delta_precise_up_to(p5, p6)


# slooooowwwwww
class Test_MGSaS_2D_T:
    n = 2.0
    cov = _cov_id(n)
    mgsas = Multivariate_GSaS_2D(cov=cov, alpha=[1.0, 1.0], k=[3.0, 4.0])
    mgsas2 = Multivariate_GSaS_2D(cov=cov, alpha=[1.1, 0.9], k=[3.0, 4.0])

    def test_peak_pdf(self):
        p1 = self.mgsas2.pdf(self.mgsas2.x0)
        p2 = self.mgsas2.pdf_at_zero()
        delta_precise_up_to(p1, p2)

    def test_id_pdf_refactor_to_t_at_x0(self):
        x0 = _x0(self.n)        
        p1 = self.mgsas.pdf(x0) 
        p2 = t(3).pdf(x0[0]) * t(4).pdf(x0[1])
        delta_precise_up_to(p1, p2)

    def test_id_pdf_refactor_to_t_at_x1(self):
        x1 = [0.2, 0.3]
        p1 = self.mgsas.pdf(x1) 
        p2 = t(3).pdf(x1[0]) * t(4).pdf(x1[1])
        delta_precise_up_to(p1, p2)

    def test_id_pdf_refactor_to_gsas_at_x1(self):
        x1 = [0.2, 0.3]
        p1 = self.mgsas2.pdf(x1) 
        p2 = gsas(alpha=1.1, k=3.0).pdf(x1[0]) * gsas(alpha=0.9, k=4.0).pdf(x1[1])
        delta_precise_up_to(p1, p2)


