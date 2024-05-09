
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
from gas_impl.unit_test_utils import *

def _x0(n): return [0.0 for x in range(int(n))]
def _cov_id(n): return np.identity(int(n))

# VERY slow stuff in this file

class Test_MGSaS_2d:
    cov = np.array([[2.1, 0.4], [0.4, 1.5]])
    alpha = 0.9
    k = 3.5
    mgsas = Multivariate_GSaS(cov=cov, alpha=alpha, k=k)

    def test_marginal_pdf_n0(self):
        x = 0.2
        p1 = self.mgsas.marginal_1d_pdf(x, 0)
        p2 = self.mgsas.marginal_pdf_2d_int(x, 0)
        delta_precise_up_to(p1, p2)

    def test_marginal_pdf_n1(self):
        x = 0.12
        p1 = self.mgsas.marginal_1d_pdf(x, 1)
        p2 = self.mgsas.marginal_pdf_2d_int(x, 1)
        delta_precise_up_to(p1, p2)


class Test_MGSaS_3d:
    cov = np.array([[2.1, 0.4, 0.2], [0.4, 1.5, 0.3], [0.2, 0.3, 1.3]])
    alpha = 0.9
    k = 4.0
    mgsas = Multivariate_GSaS(cov=cov, alpha=alpha, k=k)
    gsas = gsas(alpha=alpha, k=k)

    def test_marginal_pdf_n0(self):
        x = 0.2
        p1 = self.mgsas.marginal_1d_pdf(x, 0)
        sd = self.cov[0,0]**0.5
        p2 = self.gsas.pdf(x/sd) / sd
        delta_precise_up_to(p1, p2)


# ---------------------------------------------------------------
class Test_MGSaS_2D:
    n = 2.0
    cov = np.array([[2.1, 0.4], [0.4, 1.5]])
    mgsas = Multivariate_GSaS_2D(cov=cov, alpha=[1.1, 0.9], k=[3.0, 4.0])

    def test_peak_pdf(self):
        p1 = self.mgsas.pdf(self.mgsas.x0)
        p2 = self.mgsas.pdf_at_zero()
        delta_precise_up_to(p1, p2)

    def no_test_marginal_pdf_n0(self):
        x = 0.2
        p1 = self.mgsas.marginal_1d_pdf(x, 0)
        p2 = self.mgsas.marginal_pdf_2d_int(x, 0)
        delta_precise_up_to(p1, p2)

    def no_test_marginal_pdf_n1(self):
        x = 0.12
        p1 = self.mgsas.marginal_1d_pdf(x, 1)
        p2 = self.mgsas.marginal_pdf_2d_int(x, 1)
        delta_precise_up_to(p1, p2)
