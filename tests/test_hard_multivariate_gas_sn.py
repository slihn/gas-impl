
# test gsas

import numpy as np
import pandas as pd
import mpmath as mp
import time
from timeit import default_timer as timer
from numpy import allclose  # type: ignore

from scipy.special import gamma
from scipy.integrate import quad
from scipy import stats
from scipy.stats import multivariate_normal, multivariate_t, t


from .multivariate_sn import Multivariate_GAS_SN, Multivariate_GAS_SN_Adp_2D
from .gas_dist import gsas
from .utils import quadratic_form
from .unit_test_utils import *


def _corr_id(n): return np.identity(int(n))

# VERY slow stuff in this file

class Test_GAS_SN_2d:
    cov = np.array([[2.1, 0.4], [0.4, 1.5]])
    beta = [0.5, -0.35]
    loc = [0.1, 0.2]
    alpha = 1.3
    k = 5.5
    mg = Multivariate_GAS_SN(cov=cov, alpha=alpha, k=k, beta=beta, loc=loc)
    mg_m1 = mg.mean()

    mg0 = Multivariate_GAS_SN(cov=mg.corr, alpha=alpha, k=k, beta=beta)  # for validation

    def test_unity(self):
        start = timer()
        p1 = 1.0         
        p2 = self.mg.moment_by_2d_int(0, 0)
        end = timer()
        print(f"Multivariate_GAS_SN.moment_by_2d_int: took {end - start:.2f} seconds, total pdf = {p2}")
        delta_precise_up_to(p1, p2)

    def test_mean_1(self):
        print_time(f"Multivariate_GAS_SN mean[1]: starts")
        p1 = self.mg_m1[0]
        p2 = self.mg.moment_by_2d_int(1, 0)
        delta_precise_up_to(p1, p2, msg_prefix="mean[0]", abstol=1e-2, reltol=1e-2)
        
    def test_mean_2(self):
        print_time(f"Multivariate_GAS_SN mean[2]: starts")
        p1 = self.mg_m1[1]
        p2 = self.mg.moment_by_2d_int(0, 1)
        delta_precise_up_to(p1, p2, msg_prefix="mean[1]", abstol=1e-2, reltol=1e-2)
        
    def _validate_var(self, g, v1, v2, p1, p2):
        # this can take up to 20 minutes, hard to converge, but we need to verify it
        q1 = g.variance[v1, v2]
        mu_z = g.mean()
        mu2 = np.outer(mu_z, mu_z) 
        print_time(f"Multivariate_GAS_SN calc_var[{v1} {v2}]: starts, beta = {g.beta}, cov = {g.cov}")
        start = timer()
        q2 = g.moment_by_2d_int(p1, p2) - mu2[v1, v2]
        end = timer()
        print(f"Multivariate_GAS_SN calc_var[{v1} {v2}]: took {end - start:.2f} seconds, q1 = {q1}, q2 = {q2}")
        delta_precise_up_to(q1, q2, msg_prefix=f"var[{v1} {v2}]", abstol=1e-3, reltol=1e-3)

    def test_mg0_var_00(self):
        self._validate_var(self.mg0, 0, 0, 2, 0)

    def test_mg0_var_11(self):
        self._validate_var(self.mg0, 1, 1, 0, 2)
        
    def test_mg0_var_01(self):
        self._validate_var(self.mg0, 0, 1, 1, 1)
        
    def test_var_00(self):
        self._validate_var(self.mg, 0, 0, 2, 0)

    def test_var_11(self):
        self._validate_var(self.mg, 1, 1, 0, 2)
        
    def test_var_01(self):
        self._validate_var(self.mg, 0, 1, 1, 1)

    def _validate_marginals(self, g, x, n):
        p1 = self.mg.marginal_1d_pdf(x, n)
        p2 = self.mg.marginal_1d_pdf_by_int(x, n)
        delta_precise_up_to(p1, p2)
        
    def test_marginal_pdf_n0(self):
        self._validate_marginals(self.mg, x=0.2, n=0)

    def test_marginal_pdf_n1(self):
        self._validate_marginals(self.mg, x=0.12, n=1)


class Test_GAS_SN_3d:
    cov = np.array([[1.2, 0.65, 0.35], [0.65, 0.8, 0.2], [0.35, 0.2, 0.5]])
    beta = np.array([0.65, 0.25, 0.10])
    loc = [0.1, 0.2, -0.1]
    alpha = 1.3
    k = 5.5
    mg = Multivariate_GAS_SN(cov=cov, alpha=alpha, k=k, beta=beta, loc=loc)
    mg_m1 = mg.mean()

    def _validate_marginals(self, g, x, n):
        p1 = self.mg.marginal_1d_pdf(x, n)
        p2 = self.mg.marginal_1d_pdf_by_int(x, n)
        delta_precise_up_to(p1, p2)
        
    def test_marginal_pdf_n0(self):
        self._validate_marginals(self.mg, x=0.2, n=0)

    def test_marginal_pdf_n1(self):
        self._validate_marginals(self.mg, x=0.12, n=1)

    def test_marginal_pdf_n2(self):
        self._validate_marginals(self.mg, x=-0.1, n=2)

