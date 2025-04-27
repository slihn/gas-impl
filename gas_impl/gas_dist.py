

import numpy as np 
import pandas as pd
import mpmath as mp
from typing import Union, Optional, List
from functools import lru_cache
from scipy.stats import rv_continuous, levy_stable
from scipy.special import gamma, erf, hyp2f1, hyp1f1, betainc
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import root_scalar
import cmath


from .wright import mp_gamma, wright_m_fn_by_levy, wright_m_fn_rescaled_by_levy, M_Wright_One_Sided
from .mellin import pdf_by_mellin
from .fcm_dist import frac_chi_mean, fcm_moment
from .gas_g_skew import *  # legacy import
from .utils import calc_stats_from_moments, is_number


# --------------------------------------------------------------------------------
def gsas_moment(n: float, alpha: float, k: float, half=False) -> float:
    if not half:
        assert float(int(n)) == float(n)
        if n % 2 != 0: return 0.0  # odd moments are zero

    assert k != 0
    n = mp.mpf(n)
    k = mp.mpf(k)
    alpha = mp.mpf(alpha)
    # we use mpf for cancellation, not for output, but you need to set the precision
    k_sigma = mp.power(abs(k), 1/mp.mpf(2)-1/alpha)  # without the /sqrt(2) thing

    if k > 0:
        a = mp_gamma((n+1)/2) / mp.sqrt(mp.pi)
        b = mp.power(2.0 / k_sigma, n) 
        c = mp_gamma((k-1)/2) / mp_gamma((k-1)/alpha) if k != 1.0 else 2/alpha
        d = mp_gamma((k-n-1)/alpha) / mp_gamma((k-n-1)/2) if k != n+1 else alpha/2
        return float(a * b * c * d)
    if k < 0:
        a = mp_gamma((n+1)/2) / mp.sqrt(mp.pi)
        b = mp.power(k_sigma, n) 
        c = mp_gamma(abs(k)/2) / mp_gamma(abs(k)/alpha) if k != 0 else 2/alpha
        d = mp_gamma((abs(k)+n)/alpha) / mp_gamma((abs(k)+n)/2) if abs(k) != -n else alpha/2
        return float(a * b * c * d)
    raise Exception(f"ERROR: k is not handled properly")


def gsas_kurtosis(alpha: float, k:float, fisher: bool=True, exact_form=False):
    # TODO should have an analytic piece too
    if not isinstance(alpha, float):
        # alpha is a list
        ans = [gsas_kurtosis(alpha1, k, fisher=fisher, exact_form=exact_form) 
               for alpha1 in alpha]  # type: ignore
        return ans[0] if len(ans) == 1 else ans

    k = float(k)
    if exact_form == True:
        s = 1.0/alpha
        a = 3.0 * (k-5)/(k-3)
        b = gamma(s*(k-5)) * gamma(s*(k-1)) / gamma(s*(k-3))**2
        kurt = a * b 
    else:
        kurt = gsas_moment(n=4.0, alpha=alpha, k=k) / gsas_moment(n=2.0, alpha=alpha, k=k)**2

    return (kurt - 3.0) if fisher else kurt


def gsas_pdf_at_zero(alpha, k) -> float:
    alpha = float(alpha)
    k = float(k)
    assert k != 0 
    sigma = np.power(abs(k), 1/2.0-1/alpha) / np.sqrt(2)  # type: ignore

    if k > 0:
        g_const = gamma((k-1)/2) / gamma((k-1)/alpha) if k != 1 else 2.0 / alpha
        return sigma / np.sqrt(2*np.pi) * g_const * gamma(k/alpha) / gamma(k/2)
    if k < 0:
        g_const = gamma((abs(k)-1)/alpha) / gamma((abs(k)-1)/2) if k != -1 else alpha / 2.0
        return 1/sigma / np.sqrt(2*np.pi) * g_const * gamma(abs(k)/2) / gamma(abs(k)/alpha)
    raise Exception(f"ERROR: k is not handled properly")


def gsas_std_pdf_at_zero(alpha, k):
    p = gsas_pdf_at_zero(alpha, k)
    var = gsas_moment(n=2.0, alpha=alpha, k=k)
    sd = var**0.5 if var >= 0 else np.NaN
    return p * sd


def gsas_characteristic_fn(x, alpha, k):
    fcm = frac_chi_mean(alpha, k)
    def fn(s):
        return norm().pdf(x/s) *  fcm.pdf(s)  # type: ignore

    p2 = quad(fn, a=0, b=np.inf, limit=1000)[0]
    return p2 * np.sqrt(2*np.pi)


# --------------------------------------------------------------------------------
def isin_feller_takayasu_diamond(alpha, theta, incl_extremal=True) -> bool:
    if not (0 <= alpha <= 2):  return False
    min_alpha = np.minimum(alpha, 2.0-alpha)
    if incl_extremal:
        return True if abs(theta) <= min_alpha else False  # F.35 of Appendix F
    else:
        return True if abs(theta) < min_alpha else False


def g_from_theta(alpha, theta):
    if isin_feller_takayasu_diamond(alpha, theta) and alpha != 0:
        return (alpha - theta) / (2.0 * alpha)
    else: 
        return np.nan


def theta_from_g(alpha, g):
    theta = alpha * (1.0 - 2.0 * g)
    if isin_feller_takayasu_diamond(alpha, theta):
        return theta
    else: 
        return np.nan


def largest_alpha_for_gamma(g):
    return np.minimum(1.0/g, 1.0/(1.0-g))


def from_s1_to_feller(alpha, beta):  # return theta
    # Appendix A.2 of Lihn(2020)
    alpha2 = alpha * np.pi/2
    theta = cmath.phase( 1.0 -1j * beta * np.tan(alpha2)) * 2/np.pi
    assert abs(theta) <= alpha
    scale = abs( 1.0 -1j * beta * np.tan(alpha2))**(-1/alpha)
    return theta, scale


def from_feller_to_s1(alpha, theta):  # return beta and scale
    if theta == 0: return 0.0, 1.0

    alpha2 = alpha * np.pi/2
    assert abs(theta) <= alpha and abs(theta) <= 2.0-alpha

    def find_beta(beta):
        return cmath.phase( 1.0 -1j * beta * np.tan(alpha2)) * 2/np.pi - theta

    if abs(theta) == alpha:
        beta = np.sign(theta) * 1.0
    else:
        sol = root_scalar(find_beta, x0=0.5 * np.sign(theta), bracket=[-1.0, 1.0])
        beta = sol.root

    scale = abs( 1.0 -1j * beta * np.tan(alpha2))**(-1/alpha)
    return beta, scale


def levy_stable_from_feller(alpha, theta):
    beta, scale = from_feller_to_s1(alpha, theta)
    return levy_stable(alpha=alpha, beta=beta, scale=scale)


# --------------------------------------------------------------------------------
class gsas_gen(rv_continuous):

    def frac_chi_mean(self, alpha, k): return frac_chi_mean(alpha, k)

    def _pdf(self, x, alpha, k, kernel_shift=0.0, *args, **kwargs):
        # handle array form
        if not isinstance(alpha, float):
            assert len(alpha) == len(x), f"ERROR: len of alpha and x"
            if len(x) == 1:  # trvial case
                return self._pdf(x[0], alpha=alpha[0], k=k[0])
            
            df = pd.DataFrame(data={'x': x, 'alpha': alpha, 'k': k})
            df['pdf'] = df.parallel_apply(lambda row: self._pdf(row['x'], alpha=row['alpha'], k=row['k']), axis=1)  # type: ignore
            return df['pdf'].tolist()

        # integral form
        assert isinstance(x, float)
        assert isinstance(alpha, float)
        assert isinstance(k, float)
        assert alpha >= 0

        chi = self.frac_chi_mean(alpha=alpha, k=k)
        
        def _kernel(s: float):
            return s * norm().pdf(s*x - kernel_shift) * chi.pdf(s)  # type: ignore

        return quad(_kernel, a=0.0, b=np.inf, limit=10000)[0]

    def _cdf(self, x, alpha, k, *args, **kwargs):
        # handle array form
        if not isinstance(alpha, float):
            assert len(alpha) == len(x), f"ERROR: len of alpha and x"
            if len(x) == 1:  # trvial case
                return self._cdf(x[0], alpha=alpha[0], k=k[0])
            
            df = pd.DataFrame(data={'x': x, 'alpha': alpha, 'k': k})
            df['cdf'] = df.parallel_apply(lambda row: self._cdf(row['x'], alpha=row['alpha'], k=row['k']), axis=1)  # type: ignore
            return df['cdf'].tolist()

        # integral form
        assert isinstance(x, float)
        assert isinstance(alpha, float)
        assert isinstance(k, float)

        def _kernel(s: float):
            return erf(s*x/np.sqrt(2)) * self.frac_chi_mean(alpha=alpha, k=k).pdf(s)  # type: ignore

        cdf1 = quad(_kernel, a=0.0, b=np.inf, limit=10000)[0]
        return cdf1*0.5 + 0.5

    def _argcheck(self, *args, **kwargs):
        # Customize the argument checking here
        alpha = args[0]
        k = args[1]
        return (
            alpha >= 0  # Allow alpha to be zero or positive
            and k != 0 
        )

    def _munp(self, n, alpha, k, *args, **kwargs):
        # https://github.com/scipy/scipy/issues/13582
        # mu = self._munp(1, *goodargs)
        return gsas_moment(n=float(n), alpha=float(alpha), k=float(k))
    

gsas = gsas_gen(name="generalized symmetric alpha-stable", shapes="alpha, k")


# --------------------------------------------------------------------------------
class gsas_squared_gen(rv_continuous):

    def _array_wrapper(self, x, alpha, k, fn):
        assert len(alpha) == len(x), f"ERROR: len of alpha and x"
        if len(x) == 1:  # trvial case
            return fn(x[0], alpha=alpha[0], k=k[0])
        
        df = pd.DataFrame(data={'x': x, 'alpha': alpha, 'k': k})  # type: ignore
        df['fn'] = df.parallel_apply(lambda row: fn(
            row['x'], alpha=row['alpha'], k=row['k']), axis=1)  # type: ignore
        return df['fn'].tolist()

    def _pdf(self, x, alpha, k, *args, **kwargs):
        # handle array form
        if not isinstance(alpha, float):
            return self._array_wrapper(x, alpha, k, self._pdf)

        # integral form
        assert isinstance(alpha, float)
        assert isinstance(k, float)
        assert alpha >= 0

        assert isinstance(x, float)
        if x <= 0: return 0.0

        x2 = np.sqrt(x)
        g = gsas(alpha=alpha, k=k)
        return g.pdf(x2) / x2  # type: ignore

    def _cdf(self, x, alpha, k, *args, **kwargs):
        # handle array form
        if not isinstance(alpha, float):
            return self._array_wrapper(x, alpha, k, self._cdf)

        # integral form
        assert isinstance(alpha, float)
        assert isinstance(k, float)
        assert alpha >= 0

        assert isinstance(x, float)
        if x <= 0: return 0.0

        x2 = np.sqrt(x)
        g = gsas(alpha=alpha, k=k)
        return g.cdf(x2) - g.cdf(-x2)  # type: ignore

    def _argcheck(self, *args, **kwargs):
        # Customize the argument checking here
        alpha = args[0]
        k = args[1]
        return (
            alpha >= 0  # Allow alpha to be zero or positive
            and k != 0  # k cannot be zero
        )

gsas_squared = gsas_squared_gen(name="gsas squared", shapes="alpha, k")


# --------------------------------------------------------------------------------
def gas_mellin_transform(s, eps: float, k: float, g: float):
    # this is the one-sided GAS without the density adjustment
    # this uses (eplison, gamma) representation
    m_wright = g * gamma(s) / gamma(1.0 - g + g*s)  # TODO need to verify Mellin transform of skew M-Wright distribution
    assert k != 0
    if k > 0:
        gamma_k = eps / g if k == 1.0 else gamma(g * (k-1.0)) / gamma(eps * (k-1.0))
        sigma_const = 1.0 / (k**(g - eps))**(s-1.0)
        gamma_ks = g / eps if k == s else gamma(eps * (k-s)) / gamma(g * (k-s)) 
        chi = sigma_const * gamma_k * gamma_ks 
        return m_wright * chi
    else:
        k = abs(k)
        gamma_k = gamma(g * k) / gamma(eps * k)
        sigma_const = (k**(g - eps) * g)**(s-1.0)  # this is different
        gamma_ks = gamma(eps * (k+s-1)) / gamma(g * (k+s-1)) 
        chi = sigma_const * gamma_k * gamma_ks 
        return m_wright * chi


def gas_moment_one_sided_from_mellin(n: float, alpha: float, k: float, theta: float) -> float:
    eps = 1.0 / alpha
    g = (alpha - theta) / (2.0 * alpha)
    return gas_mellin_transform(float(n) + 1.0, eps, float(k), g) 


# LihnStable is a class to handle the stable PDF more elegantly
# lihn_stable us a bit awkward to deal with the reflective nature of the stable distribution
class LihnStable:
    slope_sigma_pow_spec_list: List[str] = ['g', 'gr', 'half', 'none', 'side', 'zero']
    
    def __init__(self, alpha, k, theta, 
                 reflected_iter: int = 0,
                 slope_sigma_pow_spec: str = 'g' # examples are: zero=(0,1) aka canonical, half=(0.5,0.5), g=(g,1-g)
                 ): 
        self.alpha = float(alpha)
        self.k = float(k)
        self.theta = float(theta)
        assert abs(self.theta) <= np.min(np.array([self.alpha, 2.0-self.alpha]))

        self.eps = 1.0 / self.alpha
        self.g = (self.alpha - self.theta) / (2.0 * self.alpha)  # instead of 0.5
        self.fcm = self.get_fcm()
        assert 0 < self.g < 1.0

        self.reflected_iter: int = reflected_iter
        self.slope_sigma_pow_spec = slope_sigma_pow_spec
        assert self.slope_sigma_pow_spec in self.slope_sigma_pow_spec_list or is_number(self.slope_sigma_pow_spec)
        # the is_number() case is primarily for the proof of invariant skewness and kurtosis
        if self.slope_sigma_pow_spec == 'g':
            self.slope_sigma_pow = self.g
        elif self.slope_sigma_pow_spec == 'gr':
            self.slope_sigma_pow = 1.0 - self.g
        elif self.slope_sigma_pow_spec == 'half':
            self.slope_sigma_pow = 0.5
        elif self.slope_sigma_pow_spec == 'side':
            self.slope_sigma_pow = 0.0 if self.g >= 0.5 else 1.0  # the lesser side gets full adjustment
        elif self.slope_sigma_pow_spec == 'none':
            self.slope_sigma_pow = 0.0  # no slope adjustment, only the density adjustment
        elif self.slope_sigma_pow_spec == 'zero':
            self.slope_sigma_pow = 0.0 if not self.is_reflected else 1.0  # 0 and 1
        elif is_number(self.slope_sigma_pow_spec):
            h = float(self.slope_sigma_pow_spec)
            self.slope_sigma_pow = h if not self.is_reflected else (1.0 - h)
            assert 0 <= self.slope_sigma_pow <= 1.0
        else:
            raise Exception(f"ERROR: slope_sigma_pow_spec is not handled: {self.slope_sigma_pow_spec}")

        self.reflected_gas: Optional[LihnStable] = None  # this is a cache for new_by_negative_theta()

    @property
    def is_reflected(self) -> bool: return self.reflected_iter % 2 != 0
    
    def new_by_negative_theta(self):
        if self.reflected_gas is None:  
            self.reflected_gas = LihnStable(self.alpha, self.k, -self.theta, 
                                            reflected_iter = self.reflected_iter + 1, 
                                            slope_sigma_pow_spec = self.slope_sigma_pow_spec)
        assert self.reflected_gas is not None
        return self.reflected_gas

    def get_fcm(self):  return frac_chi_mean(alpha=self.alpha, k=self.k, theta=self.theta)

    @lru_cache(maxsize=2)
    def get_fcm_moment(self, n: int): return fcm_moment(n, self.alpha, self.k, theta=self.theta)

    def get_total_slope_sigma_pow(self):  return self.slope_sigma_pow + self.new_by_negative_theta().slope_sigma_pow
    
    @lru_cache(maxsize=2)
    def unadjusted_pdf_at_zero(self): 
        return self.g**(1-self.g) / gamma(1-self.g) * self.get_fcm_moment(1)

    @lru_cache(maxsize=2)
    def unadjusted_slope_at_zero(self):
        # S_{alpha,k}^\theta 
        return -self.g**(1-2*self.g) / gamma(1-2*self.g) * self.get_fcm_moment(2)

    @lru_cache(maxsize=2)
    def _get_pdf_adj_factor(self): 
        p1 = self.unadjusted_pdf_at_zero()
        p2 = self.new_by_negative_theta().unadjusted_pdf_at_zero()
        return p1 / p2

    @lru_cache(maxsize=2)
    def _get_slope_adj_factor(self):
        if self.g == 0.5: return 1.0
        s1 = self.unadjusted_slope_at_zero()
        s2 =  -1.0 * self.new_by_negative_theta().unadjusted_slope_at_zero()
        return s1 / s2
    
    @lru_cache(maxsize=2)
    def get_slope_sigma(self):
        # \Sigma_{alpha,k}^\theta, aka \Sigma^+
        return self._get_slope_adj_factor() / self._get_pdf_adj_factor() 

    # ----------------------------------------------------------------------
    # Canonical definitions. They have to be evaluated from the non-reflected side
    @lru_cache(maxsize=2)
    def Sigma(self) -> float:
        if self.is_reflected:  return self.new_by_negative_theta().Sigma()
        # \Sigma^- in the canonical definition
        return self._get_pdf_adj_factor() / self._get_slope_adj_factor()
    
    @lru_cache(maxsize=2)
    def Psi(self) -> float:
        if self.is_reflected:  return self.new_by_negative_theta().Psi()
        # \Psi in the canonical definition
        return self._get_pdf_adj_factor()**2 / self._get_slope_adj_factor()

    @lru_cache(maxsize=2)
    def A_plus(self) -> float:
        if self.is_reflected:  return self.new_by_negative_theta().A_plus()
        # A^+ in the canonical definition
        return self.g + self.Psi() * (1.0 - self.g)

    @lru_cache(maxsize=2)
    def A_minus(self) -> float:
        if self.is_reflected:  return self.new_by_negative_theta().A_minus()
        # A^- in the canonical definition
        return self.A_plus() / self.Psi() 


    # ----------------------------------------------------------------------
    @lru_cache(maxsize=2)
    def get_adj_side_density(self):
        # D_{alpha,k}^\theta 
        return self.g * self.get_slope_sigma()**self.slope_sigma_pow / self.unadjusted_pdf_at_zero()

    @lru_cache(maxsize=2)
    def get_total_density_adj_factor(self):
        return self.get_adj_side_density() + self.new_by_negative_theta().get_adj_side_density()

    @lru_cache(maxsize=2)
    def get_normalized_side_density(self):
        return self.get_adj_side_density() / self.get_total_density_adj_factor()

    @lru_cache(maxsize=2)
    def get_adj_m_wright_rescaling_factor(self):
        return self.g**self.g * self.get_slope_sigma()**self.slope_sigma_pow

    def pdf(self, x: float) -> float:
        if x < 0:
            return self.new_by_negative_theta().pdf(-x)

        # -----------------------------------------------------------------------------------------
        # the main part
        g3 = self.get_adj_m_wright_rescaling_factor()
        assert x >= 0
        
        def _m_wr_g3(x): 
            return float(wright_m_fn_by_levy(x/g3, alpha=self.g)) / g3  # type: ignore

        def _kernel(s: float):
            return s * _m_wr_g3(s*x) * self.fcm.pdf(s)  # type: ignore

        p = quad(_kernel, a=0.0, b=np.inf, limit=10000)[0]
        p = p * self.get_normalized_side_density()
        return p
    
    def cdf(self, x: float) -> float:
        p2 = self.new_by_negative_theta().get_normalized_side_density()

        if x < 0:
            p = self.new_by_negative_theta().cdf(-x)
            return 1.0 - p

        # -----------------------------------------------------------------------------------------
        # the main part
        g3 = self.get_adj_m_wright_rescaling_factor()
        message = f"for x={x}, alpha={self.alpha}, k={self.k}, theta={self.theta}, g={self.g}, slope_sigma_pow_spec={self.slope_sigma_pow_spec}"

        # m_wr = mainardi_wright_fn_in_gsc(self.g)  # this is very slow
        m_wr = M_Wright_One_Sided(self.g)
        assert x >= 0
        def _kernel(s: float):  return m_wr.cdf(s*x/g3) * self.fcm.pdf(s)  # type: ignore

        p = quad(_kernel, a=0.0, b=np.inf, limit=10000)[0]

        p = p * self.get_normalized_side_density()
        return p + p2

    def pdf_unadjusted(self, x: float) -> float:
        if x < 0:
            return self.new_by_negative_theta().pdf_unadjusted(-x)

        # -----------------------------------------------------------------------------------------
        assert x >= 0
        def _kernel(s: float):  return s * wright_m_fn_rescaled_by_levy(x*s, g=self.g) * self.fcm.pdf(s)  # type: ignore
        p = quad(_kernel, a=0.0, b=np.inf, limit=10000)[0]
        return p

    def pdf_by_mellin(self, x):
        # this is also unadjusted
        if x == 0: return np.NAN 
        if x < 0:  return self.new_by_negative_theta().pdf_by_mellin(-x)
        assert x >= 0
        return pdf_by_mellin(x, lambda s: gas_mellin_transform(s, self.eps, self.k, self.g))

    @lru_cache(maxsize=10)
    def original_gas_moment_one_sided(self, n) -> float:
        one = mp.mpf(1)
        n = mp.mpf(n)
        alpha = mp.mpf(self.alpha)
        k = mp.mpf(self.k)
        theta = mp.mpf(self.theta)
        assert k != 0

        g = (alpha - theta) / (mp.mpf(2) * alpha)  # instead of 0.5
        eps = mp.mpf(1)/alpha
        if k >= 0:
            A = g * mp.power(k, -(g-eps)*n) * mp_gamma(n + one) / mp_gamma(one + g*n)
            B = eps/g if k == one else mp_gamma(g * (k-one)) / mp_gamma(eps * (k-one))
            C = g/eps if k == n + one else mp_gamma(eps * (k-n-one)) / mp_gamma(g * (k-n-one))
            return float(A * B * C)
        else:
            k = abs(k)
            A = mp.power(g, n+1) * mp.power(k, (g-eps)*n) * mp_gamma(n+1) / mp_gamma(1+g*n)
            B = mp_gamma(g * k) / mp_gamma(eps * k)
            C = mp_gamma(eps * (k+n)) / mp_gamma(g * (k+n))
            return float(A * B * C)

    def moment_one_sided(self, n) -> float:
        A = self.get_normalized_side_density()
        B = self.get_slope_sigma()**(n * self.slope_sigma_pow) / self.g 
        C = self.original_gas_moment_one_sided(n)
        return A * B * C

    @lru_cache(maxsize=10)
    def moment(self, n) -> float:
        return self.moment_one_sided(n) + self.new_by_negative_theta().moment_one_sided(n) * (-1.0)**n

    @lru_cache(maxsize=10)
    def moment_unadjusted(self, n) -> float:
        return self.original_gas_moment_one_sided(n) + self.new_by_negative_theta().original_gas_moment_one_sided(n) * (-1.0)**n

    @lru_cache(maxsize=2)
    def calc_stats_from_moments(self, unadjusted: bool = False):
        moments = pd.Series({
            n: (self.moment(n) if not unadjusted else self.moment_unadjusted(n)) 
            for n in range(5)
        })
        return calc_stats_from_moments(moments)

    @lru_cache(maxsize=2)
    def mean(self):
        return self.calc_stats_from_moments()['mean']

    @lru_cache(maxsize=2)
    def variance(self):
        return self.calc_stats_from_moments()['var']
    
    @lru_cache(maxsize=2)
    def skewness(self):
        return self.calc_stats_from_moments()['skew']
    
    @lru_cache(maxsize=2)
    def kurtosis(self):
        # this is excess kurtosis (normal kurtosis is 0.0)
        return self.calc_stats_from_moments()['kurtosis']


class lihn_stable_gen(rv_continuous):

    def frac_chi_mean(self, alpha, k, theta): return frac_chi_mean(alpha, k, theta)

    def _pdf(self, x, alpha, k, theta, *args, **kwargs):
        # handle array form
        if not isinstance(alpha, float):
            assert len(alpha) == len(x), f"ERROR: len of alpha and x"
            if len(x) == 1:  # trvial case
                return self._pdf(x[0], alpha=alpha[0], k=k[0], theta=theta[0])
            
            df = pd.DataFrame(data={'x': x, 'alpha': alpha, 'k': k, 'theta': theta})
            df['pdf'] = df.parallel_apply(lambda row: self._pdf(row['x'], alpha=row['alpha'], k=row['k'], theta=row['theta']), axis=1)  # type: ignore
            return df['pdf'].tolist()
            # return [self._pdf(x1, df=df1, alpha=a1) for x1, df1, a1 in zip(x, df, alpha)]

        # integral form
        assert isinstance(x, float)
        assert isinstance(alpha, float)
        assert isinstance(k, float)
        assert isinstance(theta, float)

        # TODO if theta == 0, it is a GSAS, mw = norm() is much faster, do we need to accelerate this?
        return LihnStable(alpha=alpha, k=k, theta=theta).pdf(x)

    def _cdf(self, x, alpha, k, theta, *args, **kwargs):
        # handle array form
        if not isinstance(alpha, float):
            assert len(alpha) == len(x), f"ERROR: len of alpha and x"
            if len(x) == 1:  # trvial case
                return self._cdf(x[0], alpha=alpha[0], k=k[0], theta=theta[0])
            
            df = pd.DataFrame(data={'x': x, 'alpha': alpha, 'k': k, 'theta': theta})
            df['cdf'] = df.parallel_apply(lambda row: self._cdf(row['x'], alpha=row['alpha'], k=row['k'], theta=row['theta']), axis=1)  # type: ignore
            return df['cdf'].tolist()

        # integral form
        assert isinstance(x, float)
        assert isinstance(alpha, float)
        assert isinstance(k, float)
        assert isinstance(theta, float)
        
        return LihnStable(alpha=alpha, k=k, theta=theta).cdf(x)

    def _argcheck(self, *args, **kwargs):
        # Customize the argument checking here
        alpha = args[0]
        k = args[1]
        theta = args[2]
        g = (alpha - theta) / (2.0 * alpha)
        return (
            alpha >= 0  # Allow alpha to be zero or positive
            and k != 0
            and abs(theta) <= np.minimum(alpha, 2.0-alpha) # F.35 of Appendix F
            and 0 <= g <= 1.0
        )

    def _munp(self, n, alpha, k, theta, *args, **kwargs):
        # https://github.com/scipy/scipy/issues/13582
        # mu = self._munp(1, *goodargs)
        return LihnStable(alpha=alpha, k=k, theta=theta).moment(n)


lihn_stable = lihn_stable_gen(name="generalized alpha-stable", shapes="alpha, k, theta")
# theta is Feller's parametrization


# --------------------------------------------------------------------------------
# M_{alpha,k}(b,c; x) from Lemma 8.3 of Lihn(2024)
def frac_hyp_fn(x, alpha, k, b, c):
    assert isinstance(x, float)
    fc = frac_chi_mean(alpha=alpha, k=k)

    def _kernel(s: float):
        m = hyp1f1(b, c, x*k* s**2/2) 
        return s * m * fc.pdf(s)  # type: ignore

    fm = quad(_kernel, a=0.0, b=np.inf, limit=10000)[0]
    return fm * np.sqrt(k / (2 * np.pi))


def t_cdf_by_hyp2f1(x, k):
    k = float(k)
    x = float(x)
    c = gamma((k+1)/2) / gamma(k/2) / np.sqrt(k*np.pi)
    f21 = hyp2f1(0.5, (k+1)/2, 1.5, -x**2/k)
    cdf_integral = 2*x * c * f21
    return 0.5 + 0.5 * cdf_integral


def t_cdf_by_betainc(x, k, use_variant=False):
    k = float(k)
    if x < 0: return 1.0 - t_cdf_by_betainc(-x, k, use_variant=use_variant)
    p = k / (x**2 + k)
    cdf_integral = (1.0 - betainc(k/2, 0.5, p)) if not use_variant else betainc(0.5, k/2, 1.0-p)
    return 0.5 + 0.5 * cdf_integral


def t_cdf_by_binom(x, k):
    k = float(k)
    if x < 0: return 1.0 - t_cdf_by_binom(-x, k)
    p = k / (x**2 + k)
    n = (k-1) / 2
    ks = k/2 - 1.0
    cdf_integral = betainc(n-ks, ks+1, 1.0-p)  # should be binom(n,p).cdf(ks), but this doesn't work since we can't make both n and ks to be integers
    assert not np.isnan(cdf_integral)
    return 0.5 + 0.5 * cdf_integral

