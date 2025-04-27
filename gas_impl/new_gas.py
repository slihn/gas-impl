
from typing import Optional
import numpy as np 
import pandas as pd
from functools import lru_cache
from scipy.special import gamma 
from scipy.integrate import quad

from .mellin import mellin_integral, mellin_integral_quad
from .fcm_dist import fcm, fcm_moment
from .wright import wright_m_fn_by_levy, wright_mainardi_fn_ts
from .gas_dist import gas_mellin_transform


# TODO this is a legacy experimental code, not for outside users
    
class NewGAS:
    def __init__(self, alpha, k, theta, scale=1.0):
        self.alpha = float(alpha)
        self.k = float(k)
        self.theta = float(theta)
        self.scale = float(scale)
        assert abs(self.theta) <= np.min(np.array([self.alpha, 2.0-self.alpha]))

        self.eps = 1.0 / self.alpha
        self.g = (self.alpha - self.theta) / (2.0 * self.alpha)  # instead of 0.5
        self.fcm = self.get_fcm()
        
        self.reflected_gas: Optional[NewGAS] = None  # this is a cache for new_by_negative_theta()

    def new_by_negative_theta(self):
        if self.reflected_gas is None:  
            self.reflected_gas = NewGAS(self.alpha, self.k, -self.theta, scale=self.scale)
        assert self.reflected_gas is not None
        return self.reflected_gas

    def pdf_by_mellin(self, x, c = 0.3, smax = 200.0, num = 1000001):
        # Note: only one side of x is correct

        if x == 0: return np.NAN 
        if x < 0:
            return self.new_by_negative_theta().pdf_by_mellin(-x, c=c, smax=smax, num=num)

        x = x / self.scale

        def _integrand(s):
            return gas_mellin_transform(s, self.eps, self.k, self.g) * x**(-s) 

        # c between 0 and 1
        assert 0.0 < c < 1.0
        # p = mellin_integral(_integrand, c=c, smax=smax, num=num) 
        p = mellin_integral_quad(_integrand, c=c, smax=smax) 
        assert np.imag(p) < 1e-8, f"ERROR: imag part of p is not zero: {np.imag(p)}"  # type: ignore
        return np.real(p) / self.scale  # type: ignore
    

    def get_fcm(self):  return fcm(alpha=self.alpha, k=self.k, theta=self.theta)

    @lru_cache(maxsize=2)
    def get_fcm_moment(self, n: int): return fcm_moment(n, self.alpha, self.k, theta=self.theta)
    
    @lru_cache(maxsize=2)
    def raw_pdf_at_zero(self): return np.sqrt(self.g) / gamma(1-self.g) * self.get_fcm_moment(1)

    @lru_cache(maxsize=2)
    def get_adj_factor(self): 
        p1 = self.raw_pdf_at_zero()
        p2 = self.new_by_negative_theta().raw_pdf_at_zero()
        return p1 / p2
    
    @lru_cache(maxsize=2)
    def get_slope_adj_factor(self):
        if self.g == 0.5: return 1.0
        s1 = -1 / gamma(1-2*self.g) * fcm_moment(2, self.alpha, self.k, theta=self.theta)
        s2 =  1 / gamma(2*self.g-1) * fcm_moment(2, self.alpha, self.k, theta=-self.theta)
        return s1 / s2
    
    @lru_cache(maxsize=2)
    def get_slope_sigma(self):
        return self.get_slope_adj_factor() / self.get_adj_factor() 

    @lru_cache(maxsize=2)
    def get_adj_side_density(self):
        return self.g * self.get_slope_sigma()**0.5 / self.raw_pdf_at_zero()

    @lru_cache(maxsize=2)
    def get_total_density_adj_factor(self):
        return self.get_adj_side_density() + self.new_by_negative_theta().get_adj_side_density()

    def pdf_by_fcm(self, x, adj=False):
        if x < 0:
            return self.new_by_negative_theta().pdf_by_fcm(-x, adj=adj)

        # -----------------------------------------------------------------------------------------
        if adj == True:  
            x = x / self.get_slope_sigma()**0.5

        x = x / self.scale
        g = self.g 
        g2 = np.sqrt(g)
        
        def _m_wr_g(x): return float(wright_m_fn_by_levy(abs(x)/g2, alpha=g)) * g2  # type: ignore

        def _kernel(s: float):
            return s * _m_wr_g(s*x) * self.fcm.pdf(s)  # type: ignore

        p = quad(_kernel, a=0.0, b=np.inf, limit=10000)[0] / self.scale
        if adj == True: 
            p = p / self.raw_pdf_at_zero() / self.get_total_density_adj_factor()
        return p

    def pdf_by_fcm_k_adjusted(self, x):
        if x < 0:
            return self.new_by_negative_theta().pdf_by_fcm_k_adjusted(-x)

        # -----------------------------------------------------------------------------------------
        x = x / self.scale
        g = self.g 
        g3 = np.sqrt(g * self.get_slope_sigma())
        
        def _m_wr_g3(x): 
            return float(wright_m_fn_by_levy(abs(x)/g3, alpha=g)) / g3  # type: ignore

        def _kernel(s: float):
            return s * _m_wr_g3(s*x) * self.fcm.pdf(s)  # type: ignore

        p = quad(_kernel, a=0.0, b=np.inf, limit=10000)[0] / self.scale
        p = p * self.get_adj_side_density() / self.get_total_density_adj_factor()
        return p

    def pdf_by_fcm_ts(self, x):
        if self.g < 0.5:
            return self.new_by_negative_theta().pdf_by_fcm_ts(-x)

        assert self.g >= 0.5
        x = x / self.scale
        g = self.g 
        g2 = np.sqrt(g)
        
        def _m_wr_g(x): return float(wright_m_fn_by_levy(x/g2, alpha=g)) * g2  # type: ignore  # or use wright_mainardi_fn_ts()

        def _kernel(s: float):
            return s * _m_wr_g(s*x) * self.fcm.pdf(s)  # type: ignore

        # b: upper bound of the integral has to be finite, since wright_mainardi_fn_ts(x,alpha) has issue for very large x
        return quad(_kernel, a=0.0, b=np.inf, limit=10000)[0] / self.scale

    def pdf(self, x):

        # TODO unfortunately _ts() is not right
        # _ts() version has issue with undefined higher moments on the side where the tail stretches far
        # this is just the fundamental thing from the extremal stable distribution
        # so we are back to square one !

        # return self.pdf_by_fcm_ts(x)

        return self.pdf_by_fcm_k_adjusted(x)


    def original_gas_moment_one_sided(self, n):
        n = float(n)
        alpha = float(self.alpha)
        k = float(self.k)
        theta = float(self.theta)

        g = (alpha - theta) / (2.0 * alpha)  # instead of 0.5
        eps = 1/alpha
        A = g * k**(-(g-eps)*n) * gamma(n+1) / gamma(1+g*n)
        B = eps/g if k == 1.0 else gamma(g * (k-1)) / gamma(eps * (k-1))
        C = g/eps if k == n+1 else gamma(eps * (k-n-1)) / gamma(g * (k-n-1))
        return A * B * C

    def moment_one_sided(self, n):
        A = self.get_adj_side_density() / self.get_total_density_adj_factor()
        B = self.get_slope_sigma()**(n/2.0) / self.g 
        C = self.original_gas_moment_one_sided(n)
        return A * B * C


    def moment(self, n):
        return self.moment_one_sided(n) + self.new_by_negative_theta().moment_one_sided(n) * (-1.0)**n
