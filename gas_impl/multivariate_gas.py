from math import isnan
import numpy as np 
import pandas as pd
from typing import List, Optional

from scipy.integrate import quad
from scipy.stats import multivariate_normal
from numpy.linalg import det, inv  # type: ignore
from numpy import diag


from .fcm_dist import frac_chi_mean, fcm_moment
from .gas_dist import gsas, lihn_stable
from .multivariate import Multivariate_Base, is_pos_def


# TODO Work-In-Progress: This stil needs a lot of theoretical work
# --------------------------------------------------------------------------------
# This is a Multivariate implementation based on the work in the lihn_stable distribution (2024)
# But in 2025, my main effort is on GAS-SN, not in this area !
# --------------------------------------------------------------------------------
class Multivariate_GAS(Multivariate_Base):
    # not sure how to do this? Elliptical distribution seems always symmetric
    def __init__(self, cov, alpha, k, theta):
        super().__init__(cov)
        self.alpha = float(alpha)
        self.k = float(k)
        self.theta = float(theta)
        self.fcm = frac_chi_mean(self.alpha, self.k, self.theta)
        self.gas_unit = lihn_stable(alpha=self.alpha, k=self.k, theta=self.theta)  

    @property
    def variance(self):
        return self.cov * fcm_moment(-2.0, self.alpha, self.k, self.theta)

    def pdf1(self, x):
        assert len(x) == self.n 
        assert isinstance(x[0], float)
        x0 = np.zeros(self.n)

        def _kernel(s: float):
            rvn = multivariate_normal(x0, cov=self.cov * s**(-2))  # type: ignore
            return self.fcm.pdf(s) * rvn.pdf(x)  # type: ignore

        return quad(_kernel, a=0.0001, b=np.inf, limit=100000)[0]

    def pdf_at_zero(self):
        return fcm_moment(self.n, self.alpha, self.k) / self.mv_norm_const

    def marginal_1d_pdf(self, x, n: int):
        x = float(x)
        assert isinstance(x, float)
        assert isinstance(n, int)
        assert n < self.n
        sd = self.cov[n,n]**0.5
        return self.marginal_1d_rv(n, scale=1.0).pdf(x/sd) / sd  # type: ignore

    def marginal_1d_rv(self, n: int, scale: Optional[float] = None):
        # if scale is None, it is taken from cov
        assert isinstance(n, int)
        assert n < self.n
        sd = self.cov[n,n]**0.5 if scale is None else scale
        return gsas(alpha=self.alpha, k=self.k, scale=sd)

