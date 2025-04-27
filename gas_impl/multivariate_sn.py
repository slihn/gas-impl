# this file implements the Azzalini's skew normal distribution, Azzalini and Capitanio (2014)
# especially for skew-t distribution and various multivariate distributions mentioned in the book

from functools import lru_cache
import pandas as pd
import numpy as np
import math
from numpy.linalg import inv, det  # type: ignore
from numpy import diag, multiply, allclose  # type: ignore

from typing import List, Union, Optional, Tuple, Sequence, TypeVar
from abc import abstractmethod

from scipy.special import gamma
from scipy.linalg import sqrtm, eigh
from scipy.integrate import quad
from scipy.stats import skewnorm, norm, t, chi2, f
from scipy.stats import multivariate_normal

from .utils import moment_by_2d_int, marginal_1d_pdf_by_int, quadratic_form, OneSided_From_RVS, nearestPD
from .fcm_dist import frac_chi_mean, fcm_moment
from .ff_dist import frac_f
from .multivariate import Multivariate_GSaS, Multivariate_T, is_pos_def, _is_list, _calc_rho
from .gas_dist import gsas
from .gas_sn_dist import SN_Std, SN, ST_Std, GAS_SN_Std, GAS_SN


SequenceFloat = Union[Sequence[float], np.ndarray]

# -------------------------------------------------------------------------------------
def _func_wrapper(col, fn, x, single_thread=False):
    if single_thread:
        rs = [fn(el) for el in x]  # list of list implementation, single thread
    else:
        df = pd.DataFrame(data=[{'x': el} for el in x])
        df[col] = df['x'].parallel_apply(lambda x: fn(x, single_thread=True))  # can not nest parallel_apply
        rs = df[col].tolist()
    return np.array(rs) if isinstance(x, np.ndarray) else rs


def cov2corr_w_arr(cov):
    w_arr = np.sqrt(np.diag(cov))  # array, extract the scale from cov
    corr = cov / np.outer(w_arr, w_arr)
    return corr, w_arr


def get_real(Q):
    assert allclose(Q.imag, 0)  # type: ignore
    return Q.real


class Multivariate_Skew_Std: 
    # The std class take corr, not cov, corr is \bar{\Omega} in the book
    # This is better aligned with how the book is written
    def __init__(self, corr: SequenceFloat, beta: SequenceFloat):
        self.corr = np.array(corr)  # type: ignore
        self.beta = np.array(beta)  # type: ignore
        self.shape = self.corr.shape
        self.n = self.shape[0]
        assert self.shape[0] == self.shape[1]
        assert self.n == len(self.beta)
        assert is_pos_def(self.corr), f"ERROR: cov is not pos_def: {self.corr}"
        assert np.diag(self.corr).prod() - 1.0 < 1e-8, f"ERROR: diag(corr) is not all 1.0: {np.diag(self.corr)}"  # type: ignore
        self.rho = _calc_rho(self.corr) if self.n == 2 else np.NAN  # only for bivariate

        self.delta = self.corr @ self.beta / np.sqrt(1 + self.beta.T @ self.corr @ self.beta)  # (5.11)
        self.b = np.sqrt(2.0 / np.pi)  # (2.27) you need to override this when needed
        self._skew_std_init = True  # this is to avoid calling the init() twice
        self._pdf_single_thread: Optional[bool] = None

    def z2_corr(self, z):
        # this is essentially the same as above when z = w_inv x
        return float(z @ inv(self.corr) @ z) 

    @lru_cache(maxsize=1)
    def beta_star(self) -> float:
        # (5.37)
        b2 = float(self.beta.T @ self.corr @ self.beta)
        assert b2 >= 0, f"ERROR: beta_star^2 = {b2} must be non-negative"
        return b2**0.5
    
    @lru_cache(maxsize=1)
    def delta_star(self) -> float:
        # (5.38)
        d2 = self.delta.T @ inv(self.corr) @ self.delta
        assert d2 >= 0, f"ERROR: delta_star^2 = {d2} must be non-negative"
        return d2**0.5

    @abstractmethod
    def _pdf1(self, x):  pass
 
    def _pdf(self, x, single_thread=False): 
        assert _is_list(x), f"ERROR: x is not a list: {type(x)}, value: {x}"
        if self._pdf_single_thread is not None:
            single_thread = self._pdf_single_thread
        if all(_is_list(el) for el in x):
            return _func_wrapper('pdf', self._pdf, x, single_thread=single_thread)

        assert len(x) == self.n
        assert isinstance(x[0], float)
        return self._pdf1(x)

    def _mean(self) -> np.ndarray:
        # (5.31) mu_z: it is an array
        return self.delta * self.b

    @abstractmethod
    def _var(self) -> np.ndarray:  pass
    
    @abstractmethod
    def _skew(self) -> float:  pass

    @abstractmethod
    def _kurtosis(self) -> float:  pass

    @abstractmethod
    def _rvs(self, size: int) -> np.ndarray:  pass

    def _moment_by_2d_int(self, p1, p2, use_mp=True, epsabs=1e-3, epsrel=1e-3):
        assert self.n == 2
        return moment_by_2d_int(self, p1=p1, p2=p2, use_mp=use_mp, epsabs=epsabs, epsrel=epsrel)

    def _marginal_beta(self, n):
        # (5.27) and (5.28) on p.130
        assert self.n in [2, 3]
        corr = self.corr
        beta = self.beta
        
        if self.n == 2:
            assert n in [0, 1]
            m = 1 - n
            den = 1.0 + beta[m]**2 * det(corr)
            b_12 = ( corr[:,n] @ beta ) / np.sqrt(den) 
            return b_12
        # ------------------------------------------------------------------
        elif self.n == 3:
            assert n in [0, 1, 2]
            Omega_22 = np.delete(np.delete(corr, n, axis=0), n, axis=1)
            Omega_21 = np.delete(corr[:, n], n)
            Omega_12 = np.delete(corr[n, :], n)  
            beta_2 = np.delete(beta, n)
            
            Omega_22_1 = Omega_22 - np.outer(Omega_21, Omega_12)
            b_12_den = 1.0 + beta_2.T @ Omega_22_1 @ beta_2
            b_12 = ( beta[n] + Omega_12 @ beta_2 ) / np.sqrt(b_12_den) 
            return b_12

        # ------------------------------------------------------------------
        raise Exception(f"ERROR: {self.n}-dimension is not supported")    

    def _quadratic_rvs(self, size):
        # generate random samples from the quadratic form
        Z = self._rvs(size)
        return quadratic_form(Z, self.corr)

    def _quadratic_rvs_mean(self, size):
        Q = self._quadratic_rvs(size)
        return np.mean(Q)

    @abstractmethod
    def _quadratic_rv(self):  pass  
    # theoretical distribution (rv) for the quadratic form
    # it is used for MLE and plotting, so it needs to have basic methods, such as pdf, cdf, ppf, rvs, etc.


class Multivariate_Skew_LocScale(Multivariate_Skew_Std):
    # this class enriches Std_Base to a location-scale family. The scale is embedded in the cov matrix
    # this base class is similar to Multivariate_Base, but simpler, since less is known for the skew cases
    # this class is not fast, since it is pretty rigorous to the detail in the book
    def __init__(self, cov, beta, loc=None):
        self.cov = np.array(cov)
        # w is Azzalini's symbol for scale
        corr, self.w_arr = cov2corr_w_arr(self.cov)
        if not hasattr(self, '_skew_std_init'):  # don't call it twice
            Multivariate_Skew_Std.__init__(self, corr=corr, beta=beta)

        self.w = np.diag(self.w_arr)  # scale matrix
        self.w_inv = inv(self.w)  # matrix
        self.w_det = det(self.w)  # float
        self.loc = np.array(loc) if loc is not None else np.zeros(self.n)
        assert len(self.loc) == self.n

    def x2_cov(self, x):
        # this is essentially the same as z2_corr(z) when z = w_inv x
        return float(x @ inv(self.cov) @ x) 

    @abstractmethod
    def pdf1(self, x):  pass
 
    def pdf(self, x, single_thread=False): 
        assert _is_list(x), f"ERROR: x is not a list: {type(x)}, value: {x}"
        if self._pdf_single_thread is not None:
            single_thread = self._pdf_single_thread
        if all(_is_list(el) for el in x):
            return _func_wrapper('pdf', self.pdf, x, single_thread=single_thread)

        assert len(x) == self.n
        assert isinstance(x[0], float)
        return self.pdf1(x)

    def std_pdf1(self, x):
        # starndard way for a location-scale family
        x = np.array(x) 
        assert isinstance(x[0], float), f"ERROR: x[0] = {x[0]} must be a float"

        z = self.w_inv @ (x - self.loc) 
        return self._pdf1(z) / self.w_det

    def rvs(self, size: int) -> np.ndarray:
        assert isinstance(size, int) and size > 0, f"ERROR: size = {size} must be a positive integer"
        Z = self._rvs(size)
        if size == 1: Z = np.array([Z])
        Y = np.array([self.w @ z + self.loc for z in Z])
        return (Y if size > 1 else Y[0])  # type: ignore
    
    def mean(self):
        return self.w @ self._mean() + self.loc
    
    def var(self):
        return self.w @ self._var() @ self.w

    @property
    def variance(self):  return self.var()  # this is for consistency with the GSaS multivariate side
    
    def var2corr(self) -> np.ndarray:
        v = self.var()
        return v / np.outer(np.diag(v)**0.5, np.diag(v)**0.5)

    def var2rho(self) -> float:
        # this is only for bivariate
        assert self.n == 2, f"ERROR: var2rho is only for 2-dim: {self.n}"
        return self.var2corr()[0,1]

    def skew(self):  return self._skew()
    
    def kurtosis(self):  return self._kurtosis()

    def moment_by_2d_int(self, p1, p2, use_mp=True, epsabs=1e-3, epsrel=1e-3):
        assert self.n == 2
        return moment_by_2d_int(self, p1=p1, p2=p2, use_mp=use_mp, epsabs=epsabs, epsrel=epsrel)

    def quadratic_rv(self):  return self._quadratic_rv()  # quadratic rv is unitless, so the two are identical
    
    def get_squared_rv(self):  return self.quadratic_rv()  # this is just an alias
    
    def quadratic_form(self, Z):
        return quadratic_form(Z, self.cov, loc=self.loc)  # type: ignore

    def canonicalizer(self, with_eig=False):  # type: ignore
        # Proposition 5.13 spectral decomposition of M
        C2 = sqrtm(self.cov)
        M = inv(C2) @ self.var() @ inv(C2)  # type: ignore
        E_raw, Q_raw = eigh(M)  # type: ignore
        # sort the eigenvalues and eigenvectors
        assert allclose(E_raw.imag, 0)  # type: ignore
        E_raw = E_raw.real
        assert np.all(E_raw > 0), f"ERROR: E_raw = {E_raw} must be non-negative"  # full-rank positive semi-definite
        idx = np.argsort(E_raw)
        Q = Q_raw[:, idx]

        E = E_raw[idx]
        # check if the eigenvalues are sorted
        assert np.all(E[0] <= E[1:]), f"ERROR: E = {E} must be sorted"
        Lambda = np.diag(E)

        H = get_real(inv(C2) @ Q)
        delta_z = H.T @ self.w @ self.delta

        if delta_z[0] < 0:
            Q[:, 0] = -Q[:, 0]
            H = get_real(inv(C2) @ Q)
            delta_z = H.T @ self.w @ self.delta
            assert delta_z[0] > 0, f"ERROR: delta_z[0] = {delta_z[0]} must be positive"
            
        # validation in the proof of Proposition 5.13
        assert allclose(inv(H), Q.T @ C2)  # type: ignore
        assert allclose(Q @ Lambda @ Q.T, M)  # type: ignore
        assert allclose(self.var(), inv(H.T) @ Lambda @ inv(H))  # type: ignore
        assert allclose(np.eye(self.n), H.T @ self.cov @ H)  # type: ignore
        # ---------------------------------------------------------------
        if not with_eig:
            return H
        else:
            return H, Lambda, Q, M  # type: ignore

    def is_canonical(self) -> bool:
        A = allclose(self.cov, np.eye(self.n))  # type: ignore
        B = self.beta[0] == self.beta_star()
        C = self.beta[1:].sum() == 0 if self.n > 1 else True
        return (A and B and C)

    def marginal_1d_pdf_by_int(self, x, n: int, use_mp=True):
        assert n in range(self.n)
        assert self.n in [2, 3]
        return marginal_1d_pdf_by_int(self, x, n, use_mp=use_mp)


# -------------------------------------------------------------------------------------
# Chapter 5 of Azzalini and Capitanio (2014)
class Multivariate_SN_Std(Multivariate_Skew_Std):
    # This is SN_d(0, \bar{\Omega}, \alpha) in Section 5.1
    def __init__(self, corr: SequenceFloat, beta: SequenceFloat):
        Multivariate_Skew_Std.__init__(self, corr=corr, beta=beta)
        self.m_norm_corr = multivariate_normal(cov=self.corr)  # type: ignore
        self.b = np.sqrt(2.0 / np.pi)  # (2.27)

    def _pdf1(self, x):
        # (5.1)
        x = np.array(x)
        assert isinstance(x[0], float), f"ERROR: x[0] = {x[0]} must be a float"
        return 2.0 * self.m_norm_corr.pdf(x) * norm.cdf(float(self.beta @ x))

    def _var(self):
        # (5.32) Sigma_z
        mu_z = self._mean()
        return self.corr - np.outer(mu_z, mu_z)  # type: ignore

    # Mardia's skewness and kurtosis in Azzalini's notation
    # when it depends on a single input such as beta_star, it is very simple
    # especially the location and scale don't matter
    def __mu_Sigma_mu(self):
        # (5.36)
        a2 = self.beta_star()**2
        b = self.b
        return b * a2 / ( 1.0 + (1.0 - b) * a2 )
    
    def _skew(self):
        # (5.39) gamma_1
        return (4 - np.pi) / 2.0 * self.__mu_Sigma_mu()**1.5

    def _kurtosis(self):
        # (5.39) gamma_2
        return 2.0 * (np.pi - 3.0) * self.__mu_Sigma_mu()**2

    def _rvs(self, size: int) -> np.ndarray:
        # (5.15), but we don't support broadcast
        assert isinstance(size, int) and size > 0, f"ERROR: size = {size} must be a positive integer"
        X0 = self.m_norm_corr.rvs(size=size)
        if size == 1:  X0 = np.array([X0])  # type: ignore
        T = norm.rvs(size=size)
        Z = np.array([ (X0[i] if (self.beta @ X0[i]) - T[i] > 0 else -X0[i]) for i in range(size)])  # type: ignore
        return (Z if size > 1 else Z[0])  # type: ignore

    def _equals(self, g: 'Multivariate_SN') -> bool:
        return (
            allclose(self.beta, g.beta) and allclose(self.corr, g.corr)
        )

    def _marginal_1d_rv(self, n: int):
        return SN_Std(self._marginal_beta(n))

    def _marginal_1d_pdf(self, x, n: int):
        return self._marginal_1d_rv(n)._pdf(x)

    def _quadratic_rv(self):
        # (5.7)
        return chi2(df=self.n)


class Multivariate_SN(Multivariate_Skew_LocScale, Multivariate_SN_Std):
    # This is SN_d(\epsilon, \Omega, \alpha) in Section 5.1
    # Note this implementation is rigorous, but not fast, more for research purpose!
    def __init__(self, cov, beta, loc=None):
        Multivariate_Skew_LocScale.__init__(self, cov=cov, beta=beta, loc=loc)
        Multivariate_SN_Std.__init__(self, corr=self.corr, beta=beta)


    def pdf1(self, x):
        # (5.3), was implemented first successfully as:
        #   y = self.beta @ self.w_inv @ (x - self.loc) 
        #   return 2.0 * self.m_norm_cov.pdf(x) * norm.cdf(y)
        return self.std_pdf1(x) 

    # def mean(self):  return self.w @ self._mean() + self.loc # (5.31)
    # def var(self):  return self.w @ Sigma_z @ self.w  # (5.32)

    def equals(self, g: 'Multivariate_SN') -> bool:
        return (
            allclose(self.beta, g.beta) 
            and allclose(self.corr, g.corr)  
            and allclose(self.cov, g.cov) 
            and allclose(self.loc, g.loc)  
        )
        
    def find_mode(self, check=False):
        # Proposition 5.14
        mode0 = SN_Std(self.beta_star())._find_mode()
        mode = self.loc + mode0 / self.delta_star() * self.w @ self.delta  # (5.49)
        if check: self.check_mode(mode)
        return mode

    def check_mode(self, mode, dx=1e-5, num_checks=1000):
        # this is run during testing, not for production
        p0 = self.pdf1(mode)

        def _generate():
            x0 = np.random.normal(size=self.n)  # type: ignore
            norm = np.linalg.norm(x0)  # type: ignore
            return x0, norm

        for i in range(num_checks):
            x0, norm = _generate()
            while norm < 1e-3:
                x0, norm = _generate()

            x = mode + x0 / norm * dx  # random point on the n-dim sphere around mode
            p1 = self.pdf1(x)
            assert p0 > p1, f"ERROR: {i}-th x={x}, pdf= {p1} is not smaller than {p0}"
     
    def marginal_1d_rv(self, n: int):
        return SN(beta=self._marginal_beta(n), scale=self.cov[n,n]**0.5, loc=self.loc[n])

    def marginal_1d_pdf(self, x, n: int):
        return self.marginal_1d_rv(n).pdf(x)
    

# ------------------------------------------------------------------------------------------
class Canonical_Base:
    def __init__(self, n: int, beta_star: float):
        self.n: int = n
        self._beta_star = float(beta_star)
        assert self.n > 1 and isinstance(self.n, int), f"ERROR: n = {self.n} must be an integer greater than 1"
        assert self._beta_star >= 0, f"ERROR: beta_star input = {self._beta_star} must be non-negative"

    def make_beta_arr(self):
        beta = np.zeros(self.n)
        beta[0] = self._beta_star
        return beta
        
        
# Section 5.1.8 Canonical form
class Cannonical_SN(Canonical_Base, Multivariate_SN):
    def __init__(self, n: int, beta_star: float):
        Canonical_Base.__init__(self, n=n, beta_star=beta_star)
        Multivariate_SN.__init__(self, cov=np.identity(n), beta=self.make_beta_arr())
    
    def sn_std_star(self):  return SN_Std(self.beta[0])
        
    def pdf_prod(self, x):
        # p.138
        x = np.array(x)
        # norm_pdf = np.prod([norm.pdf(x[i]) for i in range(self.n)])
        norm_pdf = np.exp(-(x @ x)/2.0) / (2.0 * np.pi)**(self.n/2.0)  # this is much faster
        return 2.0 * norm_pdf * norm.cdf(self._beta_star * x[0])
    
    def pdf_2parts(self, x):
        # for testing purpose
        cov1 = np.eye(self.n-1)
        return (
            self.sn_std_star()._pdf(x[0]) 
            * multivariate_normal(cov=cov1).pdf(x[1:])  # type: ignore
        )
    
    def find_mode(self):
        mode0 = self.sn_std_star()._find_mode()
        mode = np.zeros(self.n)
        mode[0] = mode0
        return mode


class Cannonical_Transform_Base:
    def __init__(self, m_sn: Multivariate_SN):
        self.m_sn: Multivariate_SN = m_sn
        self.n = self.m_sn.n
        self._beta_star: float = np.nan
        self._delta_star: float = np.nan
        self._delta_z: np.ndarray = np.repeat(np.nan, self.n)
        self._beta_z: np.ndarray = np.repeat(np.nan, self.n)
        self.H: np.ndarray = self.sn_canonicalizer()  # this has side effect to set _beta_star and _delta_star
     
    def sn_canonicalizer(self):
        # Note: loc in g doesn't go into the canonical, Z^* = H (Y - loc)
        g = self.m_sn        
        H: np.ndarray = g.canonicalizer()  # type: ignore

        corr_z = np.identity(g.n)
        self._delta_z = H.T @ g.w @ g.delta
        self._beta_z = (1.0 - self._delta_z.T @ inv(corr_z) @ self._delta_z )**-0.5 * inv(corr_z) @ self._delta_z  # (5.12) but on Z
        self._delta_star = (self._delta_z @ self._delta_z)**0.5
        self._beta_star  = self._delta_star / (1 - self._delta_star**2)**0.5
        assert allclose(self._delta_z[1:], np.zeros(g.n-1))
        assert allclose(self._beta_z[1:], np.zeros(g.n-1))
        return H

    def x2z(self, x):
        return self.H.T @ (x - self.m_sn.loc)

    @abstractmethod    
    def pdf_by_can(self, x, use_pdf_prod=True):  pass  # leave this to the implementation for now


class Cannonical_SN_Transform(Cannonical_SN, Cannonical_Transform_Base):
    def __init__(self, m_sn: Multivariate_SN):
        Cannonical_Transform_Base.__init__(self, m_sn=m_sn)
        Cannonical_SN.__init__(self, n=self.m_sn.n, beta_star=self._beta_star)
    
    def pdf_by_can(self, x, use_pdf_prod=True):
        # this handle g.loc properly
        # use_pdf_prod = True for speed
        pdf = self.pdf_prod(self.x2z(x)) if use_pdf_prod else self.pdf(self.x2z(x))
        return pdf * abs(det(self.H))


# -------------------------------------------------------------------------------------
# (6.23) of Azzalini and Capitanio (2014)
# ST is validated with GAS_SN
class Multivariate_ST(Multivariate_Skew_LocScale):
    def __init__(self, cov, k, beta, loc=None):
        Multivariate_Skew_LocScale.__init__(self, cov=cov, beta=beta, loc=loc)
        self.k = float(k)
        self.t = t(df=self.k + self.n)
        self.mt = Multivariate_T(cov=self.corr, k=self.k)
        self.b = ST_Std(k=self.k, beta=self.beta_star()).b  # below (6.25)

    def get_cdf_param(self, z):
        Q = self.z2_corr(z)  # identical to self.x2_cov(x)
        C = (self.k + self.n) / (self.k + Q) 
        return float(self.beta @ z) * C**0.5
                
    def pdf1(self, x):
        x = np.array(x) - self.loc  # TODO this is messy
        z = self.w_inv @ x
        y = self.get_cdf_param(z)
        return 2.0 * self.mt.pdf1(z) / self.w_det * self.t.cdf(y) 

    def pdf_via_gsas(self, x):
        # the following is the same formula, but using GSaS to express it
        x = np.array(x) - self.loc  # TODO this is messy
        z = self.w_inv @ x
        g_1 = Multivariate_GSaS(cov=self.cov, alpha=1.0, k=self.k)
        g_2 = gsas(alpha=1.0, k=self.k + self.n)
        y = self.get_cdf_param(z)
        return 2.0 * g_1.pdf1(x) * g_2.cdf(y)  # type: ignore

    def _rvs(self, size: int) -> np.ndarray:
        assert isinstance(size, int) and size > 0, f"ERROR: size = {size} must be a positive integer"
        Z = Multivariate_SN_Std(corr=self.corr, beta=self.beta)._rvs(size)
        if size == 1:  Z = np.array([Z])  # type: ignore
        V = chi2(df=self.k, scale=1/self.k).rvs(size=size)
        X = np.array([ z / np.sqrt(v) for z, v in zip(Z, V) ])
        return (X if size > 1 else X[0])  # type: ignore

    def _mean(self):  return self.b * self.delta  # (6.25)
    
    def _var(self):  
        # (6.26)
        assert self.k > 2.0, f"ERROR: k = {self.k} must be greater than 2.0"
        mu_z = self._mean()
        return self.k / (self.k - 2) * self.corr - np.outer(mu_z, mu_z)  # type: ignore

    # don't intend to do Std version 
    def _pdf1(self, x):  return np.nan  # not implemented
    def _skew(self) -> float:  return np.nan  # not implemented
    def _kurtosis(self) -> float:  return np.nan  # not implemented

    def _quadratic_rv(self):
        # (6.30)
        return f(self.n, self.k)  # this is a special case of frac_f when alpha=1
 

# -----------------------------------------------------------------------
# Stephen Lihn's contribution, it extends Multivariate_GSaS with skewness
# -----------------------------------------------------------------------
class Multivariate_GAS_SN_Std(Multivariate_Skew_Std):
    # elliptical distribution with skewness
    def __init__(self, corr, alpha: float, k: float, beta):
        Multivariate_Skew_Std.__init__(self, corr=corr, beta=beta)
        self.alpha = float(alpha)
        self.k = float(k)
        self.fcm = frac_chi_mean(self.alpha, self.k)
        self.m_norm_corr = multivariate_normal(cov=self.corr)  # type: ignore
        self.b = np.sqrt(2.0 / np.pi) * fcm_moment(-1, alpha=self.alpha, k=self.k)

    def _pdf1(self, x):
        x = np.array(x)
        assert len(x) == self.n 
        assert isinstance(x[0], float)
        
        def _kernel(s: float):
            p = self._pdf_sn_kernel(x, s)
            return self.fcm.pdf(s) * p  # type: ignore

        return quad(_kernel, a=0.0001, b=np.inf, limit=100000)[0]

    def _pdf_sn_kernel(self, z, s):
        # this is a local method only for the _kernel above
        zs = z * s
        y = float(self.beta @ zs)
        pdf = self.m_norm_corr.pdf(zs) * s**self.n   # pdf/self.w_det equals to N(cov/s^2).pdf(x) 
        return 2.0 * pdf * norm.cdf(y)

    def _rvs(self, size: int) -> np.ndarray:
        assert isinstance(size, int) and size > 0, f"ERROR: size = {size} must be a positive integer"
        Z = Multivariate_SN_Std(corr=self.corr, beta=self.beta)._rvs(size)
        if size == 1:  Z = np.array([Z])
        V = self.fcm.rvs(size=size)
        X = np.array([ z / v for z, v in zip(Z, V) ])
        return (X if size > 1 else X[0])  # type: ignore

    def _rvs_v2(self, size):
        V = self.fcm.rvs(size=size)
        # This is for proof of concept, quite slow due to repeated creation of MSN instance that is only used once 
        def _sn(v):  return Multivariate_SN(cov=self.corr / v**2, beta=self.beta).rvs(1)
        X = np.array([ _sn(v) for v in V ])
        return (X if size > 1 else X[0])  # type: ignore

    def _moment(self, m):
        if m == 1:  
            return self.delta * self.b # below (6.17)

        fcm_mm = fcm_moment(-m, alpha=self.alpha, k=self.k)
        if m == 2:
            return multiply(self.corr, fcm_mm)  # type: ignore
        raise Exception(f"ERROR: moment m = {m} is not supported")
        
    def _mean(self):
        return self._moment(1)
    
    def _var(self):
        m1 = self._mean()
        return self._moment(2) - np.outer(m1, m1)  # below (6.18) 

    def _skew(self) -> float:  return np.nan  # not implemented
    def _kurtosis(self) -> float:  return np.nan  # not implemented

    def _marginal_1d_rv(self, n: int):
        return GAS_SN_Std(self.alpha, self.k, beta=self._marginal_beta(n))

    def _marginal_1d_pdf(self, x, n: int):
        return self._marginal_1d_rv(n)._pdf(x)

    def _quadratic_rv(self):
        return frac_f(alpha=self.alpha, d=self.n*1.0, k=self.k)


class Multivariate_GAS_SN(Multivariate_Skew_LocScale, Multivariate_GAS_SN_Std):
    # elliptical distribution with skewness
    def __init__(self, cov, alpha: float, k: float, beta, loc=None):
        Multivariate_Skew_LocScale.__init__(self, cov=cov, beta=beta, loc=loc)
        Multivariate_GAS_SN_Std.__init__(self, corr=self.corr, alpha=alpha, k=k, beta=beta)

    def pdf1(self, x):
        return self.std_pdf1(x)

    def marginal_1d_rv(self, n: int):
        return GAS_SN(self.alpha, self.k, beta=self._marginal_beta(n), scale=self.cov[n,n]**0.5, loc=self.loc[n])

    def marginal_1d_pdf(self, x, n: int):
        return self.marginal_1d_rv(n).pdf(x)


class Cannonical_GAS_SN(Multivariate_GAS_SN, Canonical_Base):
    def __init__(self, n: int, alpha: float, k: float, beta_star: float):
        Canonical_Base.__init__(self, n=n, beta_star=beta_star)
        Multivariate_GAS_SN.__init__(self, cov=np.identity(n), alpha=alpha, k=k, beta=self.make_beta_arr())

    def pdf_prod(self, x):
        x = np.array(x)
        assert len(x) == self.n 
        assert isinstance(x[0], float)
        
        x_abs = np.sqrt(x @ x)
        if x_abs < 1e-8:  # avoid division by zero
            return (2 * np.pi)**(-self.n/2.0) * self.fcm.moment(self.n)

        bx = self.beta_star() * x[0] / x_abs  # x_abs can not be zero!
        C = (2 * np.pi)**(-(self.n-1)/2.0)
        
        def _kernel(s: float):
            # V1.1 norm_pdf = math.prod([norm.pdf(x[i] * s)*s for i in range(self.n)])
            # v1.2 norm_pdf = np.exp(-(x_abs * s)**2/2.0) * (s / np.sqrt(2.0 * np.pi))**self.n  # this is much faster
            # V1 sn_pdf = 2.0 * norm_pdf * norm.cdf(self._beta_star * x[0] * s)
            sn_pdf = skewnorm.pdf(x_abs * s, bx) * s**self.n * C  # V2 most efficient method
            return self.fcm.pdf(s) * sn_pdf  # type: ignore

        return quad(_kernel, a=0.0, b=np.inf, limit=100000)[0]


class Cannonical_GAS_SN_Transform(Cannonical_GAS_SN, Cannonical_Transform_Base):
    def __init__(self, m_gas_sn: Multivariate_GAS_SN):
        self.m_gas_sn = m_gas_sn
        m_sn = Multivariate_SN(cov=self.m_gas_sn.cov, beta=self.m_gas_sn.beta, loc=self.m_gas_sn.loc)
        Cannonical_Transform_Base.__init__(self, m_sn=m_sn)
        Cannonical_GAS_SN.__init__(self, n=self.m_sn.n, alpha=self.m_gas_sn.alpha, k=self.m_gas_sn.k, beta_star=self._beta_star)

    def pdf_by_can(self, x, use_pdf_prod=True):
        # this handle g.loc properly
        # use_pdf_prod = True for speed
        pdf = self.pdf_prod(self.x2z(x)) if use_pdf_prod else self.pdf(self.x2z(x))
        return pdf * abs(det(self.H))


# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# Adpative version
class Multivariate_GAS_SN_Std_Adp(Multivariate_Skew_Std):
    def __init__(self, corr, alpha: SequenceFloat, k: SequenceFloat, beta: SequenceFloat):
        Multivariate_Skew_Std.__init__(self, corr=corr, beta=beta)
        # this base class doesn't have limitation on self.n
        self.alpha = np.array(alpha)  # type: ignore
        self.k = np.array(k)  # type: ignore
        assert len(self.alpha) == self.n
        assert len(self.k) == self.n

        self.fcm_list = [frac_chi_mean(alpha=self.alpha[i], k=self.k[i]) for i in range(self.n)]
        self.gas_sn_std_list = [GAS_SN_Std(alpha=self.alpha[i], k=self.k[i], beta=self.beta[i]) for i in range(self.n)]
        self.m_sn = Multivariate_SN(cov=self.corr, beta=self.beta)
        self.b = np.sqrt(2.0 / np.pi) * self.fcm_moment_arr(-1)
        self._skew_std_adp_init = True  # this is to avoid calling the init() twice
        self._pdf_single_thread = True  # Adp already uses multiprocessing

    def _pdf1_integrand(self, x: np.ndarray, s: np.ndarray):
        fcm_pdfs = np.array([self.fcm_list[i].pdf(s[i]) * s[i] for i in range(self.n)])   # type: ignore
        sn_pdf = self.m_sn.pdf1(np.multiply(x, s))  # type: ignore
        return  np.prod(fcm_pdfs) * sn_pdf

    @lru_cache(maxsize=10)
    def fcm_moment_arr(self, m):
        return np.array([ fcm_moment(m, alpha=self.alpha[i], k=self.k[i]) for i in range(self.n) ])

    def _rvs(self, size):
        Z = Multivariate_SN_Std(corr=self.corr, beta=self.beta)._rvs(size)
        V = np.array([ c.rvs(size=size) for c in self.fcm_list ]).T
        return np.array([ z / v for z, v in zip(Z, V) ])

    def _rvs_v2(self, size):
        V = np.array([ c.rvs(size=size) for c in self.fcm_list ]).T
        def _sn(v):
            v_inv = inv(diag(v))
            cov = nearestPD(v_inv @ self.corr @ v_inv)
            return Multivariate_SN(cov=cov, beta=self.beta).rvs(1)
        X = np.array([ _sn(v) for v in V ])
        return (X if size > 1 else X[0])  # type: ignore

    def _moment(self, m):
        if m == 1:  
            return multiply(self.delta, self.b) # type: ignore

        if m == 2:
            mm1 = self.fcm_moment_arr(-1)
            mm2 = self.fcm_moment_arr(-2)
            
            def _mnt(i,j):
                if i == j: return mm2[i] * self.corr[i,i]
                return mm1[i] * mm1[j] * self.corr[i,j]
            
            return np.array([
                [ _mnt(i,j) for j in range(self.n) ] 
                for i in range(self.n)
            ])

        raise Exception(f"ERROR: moment m = {m} is not supported")
        
    def _mean(self):
        return self._moment(1)
    
    def _var(self):
        m1 = self._mean()
        return self._moment(2) - np.outer(m1, m1)  # below (6.18) 

    def _skew(self) -> float:  return np.nan  # not implemented
    def _kurtosis(self) -> float:  return np.nan  # not implemented

    def _quadratic_rv(self):
        # TODO the closed form isn't quite right yet
        # we use _rvs() to generate an interpolation version for MLE and plotting
        size = 1000 * 1000
        Q = self._quadratic_rvs(size)
        return OneSided_From_RVS(Q)


class Multivariate_GAS_SN_Std_Adp_2D(Multivariate_GAS_SN_Std_Adp):
    def __init__(self, corr, alpha: SequenceFloat, k: SequenceFloat, beta: SequenceFloat):
        Multivariate_GAS_SN_Std_Adp.__init__(self, corr=corr, alpha=alpha, k=k, beta=beta)
        assert self.n == 2
        
    def _pdf1(self, x, use_mp=True):
        assert len(x) == self.n 
        assert isinstance(x[0], float)
        # this is only for 2D
        return moment_by_2d_int(self, p1=0, p2=0, integrand_x=x, use_mp=use_mp)


class Multivariate_GAS_SN_Adp_2D(Multivariate_Skew_LocScale, Multivariate_GAS_SN_Std_Adp_2D):
    def __init__(self, cov, alpha: SequenceFloat, k: SequenceFloat, beta: SequenceFloat, loc=None):
        Multivariate_Skew_LocScale.__init__(self, cov=cov, beta=beta, loc=loc)
        Multivariate_GAS_SN_Std_Adp_2D.__init__(self, corr=self.corr, alpha=alpha, k=k, beta=beta)
        
    def pdf1(self, x):
        return self.std_pdf1(x)

    def marginal_1d_rv(self, n: int):
        return GAS_SN(self.alpha[n], self.k[n], beta=self._marginal_beta(n), scale=self.cov[n,n]**0.5, loc=self.loc[n])

    def marginal_1d_pdf(self, x, n: int):
        return self.marginal_1d_rv(n).pdf(x)


# -------------------------------------------------------------------------------------
# We only implment 2D for now, 3D is too complicated and too compute intensive to validate
# However, the canonical form is generic for any n

class Cannonical_GAS_SN_Std_Adp(Multivariate_GAS_SN_Std_Adp, Canonical_Base):
    # this provide a generic framework without limitation of n
    def __init__(self, alpha: SequenceFloat, k: SequenceFloat, beta_star: float):
        Canonical_Base.__init__(self, n=len(alpha), beta_star=beta_star)
        corr = np.identity(self.n)
        if not hasattr(self, '_skew_std_adp_init'):  # don't call it twice
            Multivariate_GAS_SN_Std_Adp.__init__(self, corr=corr, alpha=alpha, k=k, beta=self.make_beta_arr())

    def pdf_prod(self, x):
        # this method is not limited by self.n
        assert len(x) == self.n 
        assert isinstance(x[0], float)
        return np.prod([self.gas_sn_std_list[i]._pdf(x[i]) for i in range(self.n)])  # type: ignore


class Cannonical_GAS_SN_Adp_2D(Cannonical_GAS_SN_Std_Adp, Multivariate_GAS_SN_Adp_2D):
    def __init__(self, alpha: SequenceFloat, k: SequenceFloat, beta_star: float):
        Cannonical_GAS_SN_Std_Adp.__init__(self, alpha=alpha, k=k, beta_star=beta_star)
        Multivariate_GAS_SN_Adp_2D.__init__(self, cov=self.corr, alpha=self.alpha, k=self.k, beta=self.beta)  # type: ignore


class Cannonical_GAS_SN_Adp_2D_Transform(Cannonical_GAS_SN_Adp_2D, Cannonical_Transform_Base):
    def __init__(self, m_gas_sn: Multivariate_GAS_SN_Adp_2D):
        self.m_gas_sn = m_gas_sn
        m_sn = Multivariate_SN(cov=self.m_gas_sn.cov, beta=self.m_gas_sn.beta, loc=self.m_gas_sn.loc)
        Cannonical_Transform_Base.__init__(self, m_sn=m_sn)
        Cannonical_GAS_SN_Adp_2D.__init__(self, alpha=self.m_gas_sn.alpha, k=self.m_gas_sn.k, beta_star=self._beta_star)

    def pdf_by_can(self, x, use_pdf_prod=True):
        # this handle g.loc properly
        # use_pdf_prod = True for speed
        pdf = self.pdf_prod(self.x2z(x)) if use_pdf_prod else self.pdf(self.x2z(x))
        return pdf * abs(det(self.H))
