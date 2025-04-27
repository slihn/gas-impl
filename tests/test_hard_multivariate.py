
# test gsas

import numpy as np
import pandas as pd
import mpmath as mp
import time
from timeit import default_timer as timer

from scipy.special import gamma
from scipy.integrate import quad
from scipy import stats
from scipy.stats import multivariate_normal, multivariate_t, t


from .multivariate import Multivariate_GSaS, Multivariate_GSaS_Adp_2D
from .gas_dist import gsas
from .unit_test_utils import *


def _corr_id(n): return np.identity(int(n))

# VERY slow stuff in this file
# ---------------------------------------------------------------
# Elliptical distribution
# ---------------------------------------------------------------
class Test_MGSaS_2d:
    cov = np.array([[2.1, 0.4], [0.4, 1.5]])
    alpha = 0.9
    k = 3.5
    mgsas = Multivariate_GSaS(cov=cov, alpha=alpha, k=k)
 
    def test_marginal_pdf_n0(self):
        x = 0.2
        p1 = self.mgsas.marginal_1d_pdf(x, 0)
        p2 = self.mgsas.marginal_1d_pdf_by_int(x, 0)
        delta_precise_up_to(p1, p2)

    def test_marginal_pdf_n1(self):
        x = 0.12
        p1 = self.mgsas.marginal_1d_pdf(x, 1)
        p2 = self.mgsas.marginal_1d_pdf_by_int(x, 1)
        delta_precise_up_to(p1, p2)

    def _validate_var(self, v1, v2, p1, p2):
        # this can take up to 20 minutes, hard to converge, but we need to verify it
        g = Multivariate_GSaS(cov=self.cov, alpha=1.3, k=4.5)  # need it to converge faster in the tails
        q1 = g.variance[v1, v2]
        formatted_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"Multivariate_GSaS calc_var[{v1} {v2}]: starts at {formatted_time}") 
        start = timer()
        q2 = g.moment_by_2d_int(p1, p2)
        end = timer()
        print(f"Multivariate_GSaS calc_var[{v1} {v2}]: took {end - start:.2f} seconds, q1 = {q1}, q2 = {q2}")
        delta_precise_up_to(q1, q2, msg_prefix=f"var[{v1} {v2}]", abstol=1e-2, reltol=1e-3)

    def test_var_00(self):
        self._validate_var(0, 0, 2, 0)

    def test_var_11(self):
        self._validate_var(1, 1, 0, 2)
        
    def test_var_01(self):
        self._validate_var(0, 1, 1, 1)

    def test_unity(self):
        start = timer()
        p1 = 1.0         
        p2 = self.mgsas.moment_by_2d_int(0, 0)
        end = timer()
        print(f"Multivariate_GSaS.moment_by_2d_int: took {end - start:.2f} seconds, total pdf = {p2}")
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
        p2 = self.gsas.pdf(x/sd) / sd  # type: ignore
        delta_precise_up_to(p1, p2)


# ---------------------------------------------------------------
# Adaptive distribution
# ---------------------------------------------------------------
class Test_MGSaS_Adp_2D:
    n = 2.0
    cov = np.array([[2.1, 0.4], [0.4, 1.5]])
    mgsas = Multivariate_GSaS_Adp_2D(cov=cov, alpha=[1.1, 0.9], k=[3.0, 4.0])

    def test_peak_pdf(self):
        p1 = self.mgsas.pdf(self.mgsas.x0)
        p2 = self.mgsas.pdf_at_zero()
        delta_precise_up_to(p1, p2)

    def no_test_marginal_pdf_n0(self):
        x = 0.2
        p1 = self.mgsas.marginal_1d_pdf(x, 0)
        p2 = self.mgsas.marginal_1d_pdf_by_int(x, 0)
        delta_precise_up_to(p1, p2)

    def no_test_marginal_pdf_n1(self):
        x = 0.12
        p1 = self.mgsas.marginal_1d_pdf(x, 1)
        p2 = self.mgsas.marginal_1d_pdf_by_int(x, 1)
        delta_precise_up_to(p1, p2)
