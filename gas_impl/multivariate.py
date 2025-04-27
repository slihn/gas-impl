from math import isnan
import numpy as np 
import pandas as pd
import mpmath as mp
from functools import lru_cache
from typing import List, Optional
from abc import abstractmethod

from scipy.special import gamma, erf
from scipy.stats import norm
from scipy.integrate import quad, dblquad, tplquad
from scipy.stats import multivariate_normal
from numpy.linalg import det, inv  # type: ignore
from numpy import diag, allclose  # type: ignore

from .utils import moment_by_2d_int, isPD
from .fcm_dist import frac_chi_mean, fcm_moment
from .gas_dist import gsas


def _is_list(x): return isinstance(x, list) or isinstance(x, np.ndarray)


def _calc_rho(cov):
    n, n2 = cov.shape
    assert n == 2
    return cov[0,1] / np.sqrt(cov[0,0] * cov[1,1])


def is_pos_def(m):
    n, n2 = m.shape
    assert n == n2
    if np.array_equal(m, m.T):  return isPD(m)
    return False  # not symmetric, use nearestPD() to fix


# --------------------------------------------------------------------------------
class Multivariate_Base:
    def __init__(self, cov):
        self.cov = np.array(cov)
        assert is_pos_def(self.cov), f"ERROR: cov is not pos_def: {self.cov}"
        n, _ = self.cov.shape  # type: ignore
        self.n = n
        self.rho = _calc_rho(self.cov) if n == 2 else np.NAN  # only for bivariate
        
        self.x0 = [0.0 for x in range(int(self.n))]
        self.cov_id = np.identity(int(n))
        self.mv_norm_const = ((2*np.pi)**self.n * det(self.cov))**0.5

    def pdf1(self, x):  pass
 
    def pdf(self, x, single_thread=False): 
        assert _is_list(x), f"ERROR: x is not a list: {type(x)}, value: {x}"
        if all(_is_list(el) for el in x):
            if single_thread:
                rs = [self.pdf(el) for el in x]  # list of list implementation, single thread
            else:
                df = pd.DataFrame(data = [{'x': el} for el in x])
                df['pdf'] = df['x'].parallel_apply(lambda x: self.pdf(x, single_thread=True))  # can not nest parallel_apply
                rs = df['pdf'].tolist()
            return np.array(rs) if isinstance(x, np.ndarray) else rs

        assert len(x) == self.n
        assert isinstance(x[0], float)
        return self.pdf1(x)

    def marginal_1d_pdf_by_int(self, x, n: int):
        x = float(x)
        assert isinstance(x, float)
        assert isinstance(n, int)
        assert self.n == 2, f"ERROR: marginal_pdf for more than {self.n} dimension dimension is not supported"
        
        x_max = 20.0

        def fn0(y): return self.pdf1([x,y])
        def fn1(y): return self.pdf1([y,x])

        def _integrate(fn):
            p1 = quad(fn, a=0, b=x_max, limit=10000)[0]
            p2 = quad(fn, a=-x_max, b=0, limit=10000)[0]
            return p1 + p2

        if n == 0:  return _integrate(fn0)
        if n == 1:  return _integrate(fn1)
        raise Exception(f"ERROR: marginal_pdf for n={n}-th dimension is not supported")

    def moment_by_2d_int(self, p1, p2, use_mp=True, epsabs=1e-3, epsrel=1e-3):
        assert self.n == 2
        return moment_by_2d_int(self, p1=p1, p2=p2, use_mp=use_mp, epsabs=epsabs, epsrel=epsrel)

    @abstractmethod
    def var(self) -> np.ndarray:  pass

    @property
    def variance(self):  return self.var()

    
class Multivariate_GSaS(Multivariate_Base):
    def __init__(self, cov, alpha, k):
        super().__init__(cov)
        self.alpha = float(alpha)
        self.k = float(k)
        self.fcm = frac_chi_mean(self.alpha, self.k)
        self.gsas_unit = gsas(alpha=self.alpha, k=self.k)  

    def var(self):
        return self.cov * fcm_moment(-2.0, self.alpha, self.k)

    def pdf1(self, x):
        assert len(x) == self.n 
        assert isinstance(x[0], float)
        x0 = np.zeros(self.n)

        def _kernel(s: float):
            rvn = multivariate_normal(x0, self.cov * s**(-2)) # type: ignore
            return self.fcm.pdf(s) * rvn.pdf(x)  # type: ignore

        return quad(_kernel, a=0.0001, b=np.inf, limit=100000)[0]  # type: ignore

    def pdf_at_zero(self):
        return fcm_moment(self.n, self.alpha, self.k) / self.mv_norm_const

    def marginal_1d_pdf(self, x, n: int):
        # n is 0-based index
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



# --------------------------------------------------------------------------------
# this is the second kind
# only a reference implementation, it is very slow
class Multivariate_GSaS_Adpative(Multivariate_Base):
    def __init__(self, cov, alpha: List[float], k: List[float]) -> None:
        super().__init__(cov)
        self.alpha: List[float] = alpha
        self.k: List[float] = k

        assert len(self.alpha) == self.n
        assert len(self.k) == self.n

        self.fcm_list = [frac_chi_mean(alpha=self.alpha[i], k=self.k[i]) for i in range(self.n)]
        self.gsas_unit = [gsas(alpha=self.alpha[i], k=self.k[i]) for i in range(self.n)]

    def fcm_moments(self, order):
        return np.array([fcm_moment(order, alpha=self.alpha[i], k=self.k[i]) for i in range(self.n)])

    def var(self):
        mm1 = self.fcm_moments(-1.0)
        mm2 = self.fcm_moments(-2.0)
        
        def _mnt(i,j):
            if i == j: return mm2[i] * self.cov[i,i]
            return mm1[i] * mm1[j] * self.cov[i,j]
        
        return np.array([
            [ _mnt(i,j) for j in range(self.n) ] 
            for i in range(self.n)
        ])

    def _pdf1_integrand(self, x: np.ndarray, s: np.ndarray):
        si = inv(diag(s))  # s != 0 nor inf, please
        cov2 = si.T @ self.cov @ si
        cov2 = (cov2 + cov2.T) / 2.0  # force it to be symmetric
        try:
            rvn = multivariate_normal(self.x0, cov2)
        except:
            min_eig = np.linalg.eigvals(cov2).min()  # type: ignore
            if min_eig > 0: return 0.0  # scipy doesn't like it, ignore these extreme cases
            raise Exception(f"ERROR: multivariate_normal failed on min_eig={min_eig}, cov={cov2}, s={s}, si={si}")
        
        ps = np.array([self.fcm_list[i].pdf(s[i]) for i in range(self.n)])   # type: ignore
        return  np.prod(ps) * rvn.pdf(x)

    def pdf_at_zero(self):
        return np.prod(self.fcm_moments(1.0)) / self.mv_norm_const

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
        return gsas(alpha=self.alpha[n], k=self.k[n], scale=sd)


# this is 2D version of the second kind
class Multivariate_GSaS_Adp_2D(Multivariate_GSaS_Adpative):
    def __init__(self, cov, alpha: List[float], k: List[float]) -> None:
        super().__init__(cov, alpha, k)
        assert self.n == 2

    def pdf1(self, x, use_mp=True):
        assert len(x) == self.n
        assert isinstance(x[0], float)
        return moment_by_2d_int(self, p1=0, p2=0, integrand_x=x, use_mp=use_mp)


# this is 3D version of the second kind
# just for reference, never tested, super slow I guess
class Multivariate_GSaS_Adp_3D(Multivariate_GSaS_Adpative):
    def __init__(self, cov, alpha: List[float], k: List[float]) -> None:
        super().__init__(cov, alpha, k)
        assert self.n == 3

    def pdf1(self, x):
        assert len(x) == self.n
        assert isinstance(x[0], float)

        s_min = 0.001
        s_max = 40.0

        def _kernel3(s1: float, s2: float, s3: float):
            s = np.array([s1, s2, s3])
            return self._pdf1_integrand(x, s)
        return tplquad(_kernel3, s_min, s_max, s_min, s_max, s_min, s_max)[0] 


# --------------------------------------------------------------------------------
# (6.9) of Azzalini and Capitanio (2014)
class Multivariate_T(Multivariate_Base):
    def __init__(self, cov, k):
        Multivariate_Base.__init__(self, cov)
        self.k = float(k)

    def _constant(self, mat) -> float:
        k = self.k
        n = self.n
        return gamma((k + n) / 2.0) / (k * np.pi)**(n/2) / gamma(k/2.0) / det(mat)**0.5

    def x2_cov(self, x):
        # this is essentially the same as z2_corr(z) when z = w_inv x
        return float(x @ inv(self.cov) @ x) 

    def pdf1(self, x):
        k = self.k
        n = self.n
        x = np.array(x)
        assert isinstance(x[0], float), f"ERROR: x[0] = {x[0]} must be a float"
        Q = self.x2_cov(x)
        C = gamma((k + n) / 2.0) / (k * np.pi)**(n/2) / gamma(k/2.0) / det(self.cov)**0.5
        A = (1.0 + Q / k)**(-(k + n) / 2.0)
        return self._constant(self.cov) * A
    
    def var(self):
        return self.cov * fcm_moment(-2.0, alpha=1.0, k=self.k)
