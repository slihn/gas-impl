
import numpy as np
import pandas as pd
import time
from timeit import default_timer as timer
from numpy import allclose  # type: ignore

from scipy.stats import multivariate_normal, skewnorm, chi2
from scipy.integrate import quad
from numpy.linalg import inv, det  # type: ignore

from .multivariate_sn import *
from .multivariate import Multivariate_GSaS, Multivariate_GSaS_Adp_2D
from .gas_sn_dist import SN, gas_sn, ST
from .unit_test_utils import *


# -----------------------------------------------------------------
# Multivariate_Adp: this file is very slow

# ---------------------------------------------------------------
class Test_GAS_SN_Adp_2D_Quadratic:
    n = 2.0
    cov = np.array([[2.1, 0.4], [0.4, 1.5]])
    beta = [0.5, -0.35]
    loc = [0.1, 0.2]

    g = Multivariate_GAS_SN_Adp_2D(cov=cov, alpha=[1.1, 0.9], k=[3.9, 5.0], beta=beta, loc=loc)

    # rvs
    size = 500000
    X1 = g._rvs(size)
    X2 = g.rvs(size)
    q_mean = g.get_squared_rv().mean()  # type: ignore

    def test_quadratic_form(self):
        Q1 = quadratic_form(self.X1, self.g.corr)
        Q2 = self.g.quadratic_form(self.X2) # location scale adj
        assert allclose(self.q_mean, np.mean(Q1), rtol=0.02)
        assert allclose(self.q_mean, np.mean(Q2), rtol=0.02)

    def test_quadratic_form_v2(self):
        # this might take 10-20 minutes
        start = timer()
        X = self.g._rvs_v2(self.size)
        Q = quadratic_form(X, self.g.corr)
        qm = np.mean(Q)
        end = timer()
        print(f"Multivariate_GAS_SN Adp rvs_v2, mean(Q): took {end - start:.2f} seconds, q1 = {self.q_mean}, q2 = {qm}")
        assert allclose(self.q_mean, qm, rtol=0.02)


# ---------------------------------------------------------------
# mainly for the moment formula

class Test_GAS_SN_Adp_2D:
    n = 2.0
    cov = np.array([[2.1, 0.4], [0.4, 1.5]])
    alpha = [0.8, 1.3]
    k = [4.5, 5.5]
    beta = [0.5, -0.35]

    loc = [0.1, 0.2]
    beta0 = [0.0, 0.0]

    g = Multivariate_GAS_SN_Adp_2D(cov, alpha, k, beta=beta, loc=loc)
    gs = Multivariate_GAS_SN_Std_Adp_2D(g.corr, alpha, k, beta)

    g_can = Cannonical_GAS_SN_Adp_2D(alpha, k, beta_star=gs.beta_star())

    def test_1st_moment_0(self):
        # this takes 120 minutes, very slow
        p1 = self.gs._mean()[0]
        p2 = self.gs._moment_by_2d_int(1, 0)
        delta_precise_up_to(p1, p2)

    def test_1st_moment_1(self):
        # this takes 120 minutes, very slow
        p1 = self.gs._mean()[1]
        p2 = self.gs._moment_by_2d_int(0, 1)
        delta_precise_up_to(p1, p2)

    def test_2nd_moment_00(self):
        # this takes 120 minutes, very slow
        p1 = self.gs._moment(2)[0,0]
        p2 = self.gs._moment_by_2d_int(2, 0)
        delta_precise_up_to(p1, p2)

    def test_2nd_moment_11(self):
        # this takes 120 minutes, very slow
        p1 = self.gs._moment(2)[1,1]
        p2 = self.gs._moment_by_2d_int(0, 2)
        delta_precise_up_to(p1, p2)

    def test_2nd_moment_01(self):
        # this takes 120 minutes, very slow
        p1 = self.gs._moment(2)[0,1]
        p2 = self.gs._moment_by_2d_int(1, 1)
        delta_precise_up_to(p1, p2)

    # --------------------------------------------
    # Canonical
    def test_1st_moment_0_can(self):
        p1 = self.g_can._mean()[0]
        p2 = self.g_can._moment_by_2d_int(1, 0)
        delta_precise_up_to(p1, p2)
        
    def test_1st_moment_0_can_formula(self):
        p1 = self.g_can._mean()[0]
        p2 = np.sqrt(2/np.pi) * self.g_can.delta_star() * self.g_can.fcm_moment_arr(-1)[0]
        delta_precise_up_to(p1, p2)

    def test_1st_moment_1_can(self):
        p1 = self.g_can._mean()[1]
        p2 = self.g_can._moment_by_2d_int(0, 1)
        assert np.allclose(p1, 0.0)  # type: ignore
        assert np.allclose(p2, 0.0)  # type: ignore

        
 