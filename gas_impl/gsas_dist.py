import numpy as np 
import pandas as pd
import mpmath as mp
from functools import lru_cache
from scipy.stats import rv_continuous
from scipy.special import gamma
from scipy.stats import norm
from scipy.integrate import quad
import warnings

from .wright import wright_fn
from .stable_count_dist import stable_count
from .fcm_dist import fcm_sigma

def mp_gamma(z):
    if z > 0: return mp.gamma(z) 
    if z < 0: return mp.pi / mp.gamma(1-z) / mp.sin(z * mp.pi)
    return mp.inf


# --------------------------------------------------------------------------------
# not meant for production use
class Wright4Ways_MP:
    def __init__(self, a, b, lam, mu):
        self.a = a
        self.b = b
        self.lam = lam
        self.mu = mu
    
    def wright1(self, n, x):
        return mp.power(x, n) / mp_gamma(n+1.0) * mp_gamma(self.a*n + self.b) / mp_gamma(self.lam*n + self.mu)
    
    def wright_fn(self, x, max_n: int=100, start: int=0):
        p = mp.mpf(0)
        for k in range(start, max_n+1):
            p = p + self.wright1(k, x)
        return p

    def wright_terms(self, x, max_n: int=100, start: int=0):
        # for regularization and debugging purposes
        return [
            self.wright1(k, x)
            for k in range(start, max_n+1)
        ]


class GSaS_Wright_MP:
    def __init__(self, alpha, k, max_n=50):
        self.alpha = mp.mpf(alpha)
        self.k = mp.mpf(k)
        self.max_n = max_n
        assert k > 0
        self.wr4_large_x = Wright4Ways_MP(a=self.alpha/2, b=self.k/2, lam=-self.alpha/2, mu=0.0)
        self.wr4_small_x = Wright4Ways_MP(a=2/self.alpha, b=self.k/self.alpha, lam=1.0, mu=self.k/2) 
        self.sigma = 2.0 / self.k**(0.5-1/self.alpha)  # this is gsas sigma, not fcm_sigma
        self.S = self.alpha * mp_gamma((self.k-1)/2) / mp_gamma((self.k-1)/self.alpha) if k != 1 else 2.0
    
    def pdf(self, x, show_terms=False):
        start = 1
        z = x / self.sigma
        c = self.S / 2 / mp.sqrt(mp.pi)
        z_wr = -mp.power(z, -self.alpha)
        poly_term = c/x * mp.power(z, -self.k+1)

        if not show_terms:
            return poly_term * self.wr4_large_x.wright_fn(z_wr, max_n=self.max_n, start=start)
        else:
            terms = self.wr4_large_x.wright_terms(z_wr, max_n=self.max_n, start=start)
            return [poly_term * y for y in terms]

    def pdf_by_small_x(self, x, show_terms=False):
        start = 0
        z = -mp.power(x/self.sigma, 2)
        c = 1/self.sigma /mp.sqrt(mp.pi) * (self.S / self.alpha)
     
        if not show_terms:
            return c * self.wr4_small_x.wright_fn(z, max_n=self.max_n, start=start)
        else:
            terms = self.wr4_small_x.wright_terms(z, max_n=self.max_n, start=start)
            return [c * y for y in terms]


# --------------------------------------------------------------------------------
# this class is primarily for development and validatio purpose
# not meant for production use
class GSaS_Wright:
    def __init__(self, alpha: float, k: float, max_n=50, start=0):
        self.alpha: float = float(alpha)
        self.k: float = float(k)
        self.max_n: int = max_n
        self.start: int = start
        assert 0 < self.alpha <= 2.0
        assert self.k >= 1.0
        self.sigma = self.k**(0.5 - 1/self.alpha) / np.sqrt(2)  # 1/np.sqrt(2.0 * self.k)
        self.g_const = gamma((k-1)/2) / gamma((k-1)/self.alpha) if k != 1 else 2.0 / self.alpha  # let's make it a constant, very annoying

    def pdf_int_wright(self, x):  # integral form using W,sc
        assert isinstance(x, float)
        k = self.k
        c = self.alpha * self.sigma * self.g_const
        sc = stable_count(alpha=self.alpha/2)

        def _kernel(z: float):
            wright = sc.pdf(z**2) * gamma(2/self.alpha+1)  # use stable count can avoid divergence issue in Wright func
            return norm().pdf(self.sigma *z*x) * z**(k-1) * wright

        return c * quad(_kernel, a=0.001, b=np.inf, limit=1000)[0]

    def pdf_wright(self, x, use_int=False):  # expand wright fn, this is a large x expansion, x >> 1
        assert isinstance(x, float)

        def _term(k):
            return self.pdf_wright_term(x, k) if use_int == False else self.pdf_wright_term_int(x, k)
        
        p = np.array([_term(k) for k in range(0, self.max_n+1)])
        return sum(p)

    def pdf_wright_term(self, x, n): 
        assert isinstance(x, float)
        alpha = float(self.alpha)
        k = float(self.k)
        c = alpha / np.sqrt(np.pi) * self.g_const
        z = np.sqrt(2) / (self.sigma * x)
        arg = max([(alpha*n+k)/2, n+1, alpha*n/2])  # max of x: gamma(x) should be capped. 
        if arg <= 120:
            term = (-z**alpha)**n * gamma(alpha*n/2+k/2) / gamma(n+1) / gamma(-alpha*n/2)
        else:
            term = 0.0
        return c / (2*x) * z**(k-1) * term

    def pdf_wright_term_int(self, x, n):
        assert isinstance(x, float)
        alpha = float(self.alpha)
        k = float(self.k)
        n = float(n)
        c = alpha * self.sigma / np.sqrt(np.pi*2) * self.g_const

        def _kernel(z):
            gm = (-1.0)**n / gamma(n+1) / gamma(-alpha*n/2)
            gauss = np.exp(-(z*x*self.sigma)**2/2)
            z_pow = z ** (alpha*n+k-1) 
            return gm * gauss * z_pow

        return c * quad(_kernel, a=0.001, b=np.inf, limit=1000)[0]

    def _valid_terms(self, terms):  # TODO I don't think this is right all the time
        # the series will diverge at some point, so we track its absolute-descending trend and cut off beyond that point
        valids = [terms[0]]
        for i in range(1, len(terms)):
            if abs(terms[i]) <= abs(terms[i-1]):
                valids.append(terms[i])
        return valids
    
    def pdf_gauss(self, x, use_int=False):  # expand gauss exponential
        assert isinstance(x, float)
        
        def _term(n):
            return self.pdf_gauss_term(x, n) if use_int == False else self.pdf_gauss_term_int(x, n)
        
        p = np.array([_term(n) for n in range(0, self.max_n+1)])
        return sum(self._valid_terms(p))

    def pdf_gauss_term(self, x, n): 
        assert isinstance(x, float)
        alpha = float(self.alpha)
        k = float(self.k)

        c = self.sigma / np.sqrt(np.pi*2) * self.g_const
        z = (self.sigma * x)**2/2
        arg = max([(2*n+k)/alpha, n+1, n+k/2])  # max of x: gamma(x) should be capped. 
        if arg <= 120:
            term = (-z)**n * gamma((2*n+k)/alpha) / gamma(n+1) / gamma(n+k/2)
        else:
            term = 0.0
        return c * term

    def pdf_gauss_term_int(self, x, n): 
        assert isinstance(x, float)
        alpha = float(self.alpha)
        k = float(self.k)
        n = float(n)
        c = self.sigma * alpha / np.sqrt(np.pi*2) * self.g_const
        sc = stable_count(alpha=alpha/2)

        def _gauss_term():
            z = -(self.sigma * x)**2 / 2.0
            return z**n / gamma(n+1)

        def _kernel(z):
            g = _gauss_term()  
            z_pow = z ** (2*n + k - 1) 
            return  g * z_pow * sc.pdf(z**2) * gamma(2/alpha+1)

        return c * quad(_kernel, a=0.001, b=np.inf, limit=1000)[0]

