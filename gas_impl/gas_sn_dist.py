
from functools import lru_cache, partial
import numpy as np
import pandas as pd
from typing import Union, Optional, List
from abc import abstractmethod

from scipy.stats import rv_continuous
from scipy.stats import norm, skewnorm, halfnorm, multivariate_normal, t, chi, chi2, f
from scipy.integrate import quad, trapezoid
from scipy.special import gamma, owens_t, log_ndtr
from scipy.optimize import minimize
from scipy.interpolate import PchipInterpolator

from .fcm_dist import (
    FracChiMean,
    frac_chi_mean,
    fcm_mellin_transform,
    fcm_moment,
    fcm_normalization_constant,
    fcm_sigma,
)
from .gas_dist import gsas, gsas_squared
from .wright import norm_mellin_transform
from .mellin import pdf_by_mellin
from .utils import calc_stats_from_moments


# --------------------------------------------------------------------------------
# extend GSAS with skew normal, this is relatively easy to do!
def shifted_gsas_pdf(x, alpha, k, kernel_shift=0.0):
    # this is for the GSAS approximation in the (suppressed) tail analysis
    chi = frac_chi_mean(alpha=alpha, k=k)
    
    def _kernel(s: float):
        return s * norm().pdf(s*x - kernel_shift) * chi.pdf(s)  # type: ignore

    return quad(_kernel, a=0.0, b=np.inf, limit=10000)[0] 


def sn_mellin_transform(s, beta):
    x = beta * s**0.5
    return 2.0 * norm_mellin_transform(s) * t(s).cdf(x)


def owens_t_mellin_transform(s, beta):
    norm_mellin = norm_mellin_transform(s + 1)
    t_mellin = t(s+1).cdf(beta * np.sqrt(s+1))
    return norm_mellin / s * (t_mellin - 0.5)


def gas_sn_mellin_transform(s, alpha, k, beta, exact_form=False):
    if exact_form == True:
        x = beta * s**0.5
        A = 2.0 * t(s).cdf(x)
        B = gsas_mellin_transform(s, alpha, k)
        return A * B
    else:
        A = sn_mellin_transform(s, beta)
        B = fcm_mellin_transform(2.0-s, eps=1.0/alpha, k=k, g=0.5)
        return A * B


def gsas_mellin_transform(s, alpha, k):
    # exxplicit form for hyp2f1 study
    assert k > 0, f"ERROR: k = {k} must be positive"
    k1 = k - 1.0
    ks = k - s
    sigma = k**(1/2-1/alpha)
    A = gamma(s/2) * gamma(k1/2) / gamma(k1/alpha) * gamma(ks/alpha) / gamma(ks/2)
    B = (2/sigma)**(s-1) / np.sqrt(np.pi) / 2
    return A * B


def gas_sn_moment_by_mellin(n, alpha, k, beta):
    def f(b):  return gas_sn_mellin_transform(float(n+1), alpha=alpha, k=k, beta=b)
    return f(beta) + (-1)**n * f(-beta)


class gas_sn_gen(rv_continuous):

    def _array_wrapper(self, x, alpha, k, beta, fn):
        assert len(alpha) == len(x), f"ERROR: len of alpha and x"
        if len(x) == 1:  # trvial case
            return fn(x[0], alpha=alpha[0], k=k[0], beta=beta[0])
        
        df = pd.DataFrame(data={'x': x, 'alpha': alpha, 'k': k, 'beta': beta})  # type: ignore
        df['fn'] = df.parallel_apply(lambda row: fn(
            row['x'], alpha=row['alpha'], k=row['k'], beta=row['beta']), axis=1)  # type: ignore
        return df['fn'].tolist()

    def _pdf(self, x, alpha, k, beta, *args, **kwargs):
        # handle array form
        if not isinstance(alpha, float):
            return self._array_wrapper(x, alpha, k, beta, self._pdf)

        # integral form
        assert isinstance(x, float)
        assert isinstance(alpha, float)
        assert isinstance(k, float)
        assert isinstance(beta, float)
        assert alpha >= 0
        
        chi = frac_chi_mean(alpha=alpha, k=k)
        # note: it is unstable to use sn = skewnorm(beta) directly, e.g. when beta = -3.5 or 0, its PDF returns NaN
        
        def _kernel(s: float):
            if np.isfinite(beta):  # type: ignore
                sn = skewnorm.pdf(s*x, beta)
            elif beta == np.inf:
                sn = halfnorm.pdf(s*x)
            elif beta == -np.inf:
                sn = halfnorm.pdf(-s*x)
            else:
                raise ValueError(f"Invalid beta value: {beta}")

            return s * sn * chi.pdf(s)  # type: ignore

        return quad(_kernel, a=0.0, b=np.inf, limit=10000)[0]

    def _cdf(self, x, alpha, k, beta, *args, **kwargs):
        # handle array form
        if not isinstance(alpha, float):
            return self._array_wrapper(x, alpha, k, beta, self._cdf)

        # integral form
        assert isinstance(x, float)
        assert isinstance(alpha, float)
        assert isinstance(k, float)
        assert isinstance(beta, float)
        assert alpha >= 0
        
        chi = frac_chi_mean(alpha=alpha, k=k)
        
        def _kernel(s: float):
            if np.isfinite(beta):  # type: ignore
                sn = skewnorm.cdf(s*x, beta)
            elif beta == np.inf:
                sn = halfnorm.cdf(s*x)
            elif beta == -np.inf:
                sn = 1.0 - halfnorm.cdf(-s*x)
            else:
                raise ValueError(f"Invalid beta value: {beta}")

            return sn * chi.pdf(s)  # type: ignore

        return quad(_kernel, a=0.0, b=np.inf, limit=10000)[0]

    def _rvs(self, alpha, k, beta, *args, **kwargs):
        size = kwargs.get('size', 1)
        alpha = float(alpha)
        k = float(k)
        beta = float(beta)
        # # (2.13) random number generation
        # One rng drives both draws -- they are independent draws off a single stream.
        # Pass random_state per call: frac_chi_mean is lru_cached, so its frozen dist is
        # shared and must never have .random_state set on it.
        rng = kwargs.get('random_state')
        z = skewnorm(beta).rvs(size, random_state=rng)
        v = frac_chi_mean(alpha=alpha, k=k).rvs(size, random_state=rng)
        if size == 1:
            return z / v  # type: ignore
        else:
            return np.array(z) / np.array(v)  # type: ignore

    def _munp(self, n, alpha, k, beta, *args, **kwargs):
        n = float(n)
        beta = float(beta)

        if np.isfinite(beta):  # type: ignore
            sn = skewnorm.moment(n, beta)
        elif beta == np.inf:
            sn = halfnorm.moment(n)
        elif beta == -np.inf:
            sn = halfnorm.moment(n) * (-1)**n
        else:
            raise ValueError(f"Invalid beta value: {beta}")

        return sn * fcm_moment(-n, alpha=float(alpha), k=float(k))

    def _argcheck(self, *args, **kwargs):
        # Customize the argument checking here
        alpha = args[0]
        k = args[1]
        beta = args[2]
        return (
            alpha >= 0  # Allow alpha to be zero or positive
            and k != 0  # k cannot be zero
            and abs(beta) >= 0  # Allow beta to be any real number
        )


gas_sn = gas_sn_gen(name="generalized alpha-stable with skew normal", shapes="alpha, k, beta")

gas_sn_squared = gsas_squared  # they are the same!


# ---------------------------------------------------------------------------------------------
# SN and ST classes are intended for research purpose
# try to capture as many analyitics as possible from the book

def pdf_sqare(pdf, y: float):
    # Y = X^2
    if y <= 0: return 0.0
    y2 = np.sqrt(y)
    return (pdf(y2) + pdf(-y2)) / (2*y2)


class Univariate_Skew_Std:
    def __init__(self, beta: float, rv_square):
        self.beta = float(beta)
        self.delta = self.beta / np.sqrt(1 + self.beta**2)  # (2.6)
        self.b = np.nan  # placeholder, override it for your PDF, used in _mean()
        self.rv_square = rv_square

    def get_squared_rv(self):
        return self.rv_square

    @abstractmethod
    def _pdf(self, x: float) -> float:  pass

    @abstractmethod
    def _cdf(self, x: float) -> float:  pass

    def _pdf_square(self, x: float) -> float:
        assert self.rv_square is not None
        return self.rv_square.pdf(x)  # type: ignore

    def _mean(self) -> float:
        # (2.26) mu_z
        return self.delta * self.b

    @abstractmethod
    def _var(self) -> float:  pass

    def _moment(self, n) -> float:
        return np.nan * n  # fake placeholder

    @lru_cache(1)
    def _stats(self):
        return calc_stats_from_moments([ self._moment(n) for n in range(5) ])

    def _stats_mvsk(self):
        s = self._stats()
        return [s['mean'], s['var'], s['skew'], s['kurtosis']]
    
    @abstractmethod
    def _skew(self) -> float:  pass
    
    def _skew_from_moments(self):
        return self._stats()['skew']  # vary naive

    @abstractmethod
    def _kurtosis(self) -> float:  pass
    
    def _kurtosis_from_moments(self):
        return self._stats()['kurtosis']  # vary naive

    @abstractmethod
    def _mode(self) -> float:  pass

    def _mode_estimate(self) -> float:
        return self._mean() - self._skew() * self._var()**0.5 / 2.0 

    def _find_mode(self, check=True) -> float:
        # find mode by numerical optimization, turn off check if you need more speed
        def neg_pdf(x):
            if isinstance(x, float):  return -self._pdf(x)  # type: ignore
            return [ -self._pdf(i) for i in x ]  # type: ignore
  
        res = minimize(neg_pdf, x0=0.0)  # minimize() can send list of x
        mode = res.x[0]
        if check: self._check_mode(mode)
        return mode

    def _check_mode(self, mode, dx=1e-5):
        p0 = self._pdf(mode)
        p1 = self._pdf(mode - dx)
        p2 = self._pdf(mode + dx)
        assert p1 < p0 and p2 < p0

    def _peak_pdf(self):
        return self._pdf(self._find_mode())


class Univariate_Skew_LocScale(Univariate_Skew_Std):
    # generic class to extend Std to location-scale family
    def __init__(self, beta: float, rv_square, scale=1.0, loc=0.0):
        if not hasattr(self, 'beta'):
            # this is to avoid doing init twice
            Univariate_Skew_Std.__init__(self, beta=beta, rv_square=rv_square)

        self.loc = float(loc)
        self.scale = float(scale)  # this is w in the book

    def pdf(self, x):
        assert isinstance(x, float), f"ERROR: x = {x} must be a float"
        z = (x - self.loc) / self.scale
        return self._pdf(z) / self.scale

    def cdf(self, x):
        assert isinstance(x, float), f"ERROR: x = {x} must be a float"
        z = (x - self.loc) / self.scale
        return self._cdf(z)

    def mgf(self, t):  return np.nan  # fake placeholder, may not exist

    def mean(self):
        return self._mean() * self.scale + self.loc
    
    def var(self):
        return self._var() * self.scale**2
    
    def skew(self):  return self._skew()
    
    def kurtosis(self):  return self._kurtosis()

    def mode(self):
        return self._mode() * self.scale + self.loc

    def find_mode(self):
        return self._find_mode() * self.scale + self.loc

    def peak_pdf(self):
        return self.pdf(self.find_mode())
    
    def stats_mvsk(self):
        return [self.mean(), self.var(), self.skew(), self.kurtosis()]


# (2.1) of Azzalini and Capitanio (2014)
class SN_Std(Univariate_Skew_Std):
    # this is called SN(0,1,alpha) in the book
    def __init__(self, beta: float):
        rv_square = chi2(1)  # Porposition 2.1 (e)
        Univariate_Skew_Std.__init__(self, beta=beta, rv_square=rv_square)
        self.b = np.sqrt(2.0 / np.pi)  # (2.27)

    def _pdf(self, x):
        assert isinstance(x, float), f"ERROR: x = {x} must be a float"
        return 2.0 * norm.pdf(x) * norm.cdf(self.beta * x)

    def _cdf(self, x):
        # (2.37)
        return norm.cdf(x) - 2.0 * owens_t(x, self.beta)

    def _mellin_transform(self, s, beta=None):
        if beta is None: beta = self.beta
        assert isinstance(beta, float)
        return sn_mellin_transform(s, beta)

    def _moment(self, n):
        def f(b):  return self._mellin_transform(float(n+1), beta=b)
        return f(self.beta) + (-1)**n * f(-self.beta)
    
    def _moment_desciptive(self, n):
        # this is discussed in the book, we need to validate the claim
        mnt = 2 * norm_mellin_transform(n+1.0)
        if n % 2 == 0:  return mnt  # even moments
        x = -1 * self.beta * (n+1)**0.5
        return mnt * (1.0 - 2.0 * t(n+1).cdf(x)) 

    def _var(self):
        # (2.26) sigma_z^2
        # note that the second moment is 1.0
        return 1.0 - self._mean()**2
    
    def _skew(self):
        # (2.28) gamma_1
        return (4.0 - np.pi) / 2.0 * self._mean()**3 / self._var()**1.5

    def _kurtosis(self):
        # (2.29) gamma_2
        return 2.0 * (np.pi - 3.0) * self._mean()**4 / self._var()**2
    
    def _mode(self):
        # (2.33) mode
        if self.beta != 0:
            term3 = -0.5 * np.sign(self.beta) * np.exp(-2.0*np.pi / np.abs(self.beta))  # type: ignore
        else:
            term3 = 0.0
        return self._mean() - self._skew() * self._var()**0.5 / 2.0 + term3

    def _rvs(self, size: int, random_state=None):
        # # (2.12a) random number generation via selection
        z = [ (x0 if self.beta * x0 > x1 else -x0) for x0, x1 in multivariate_normal(cov=np.identity(2)).rvs(size=size, random_state=random_state)]  # type: ignore
        return np.array(z) if size > 1 else z[0]

    def _rvs_v2(self, size: int):
        # # (2.13) random number generation via selection and correlation
        corr = np.array([[1.0, self.delta], [self.delta, 1.0]])
        z = [ (x0 if x1 > 0 else -x0) for x0, x1 in multivariate_normal(cov=corr).rvs(size=size)]  # type: ignore
        return np.array(z) if size > 1 else z[0]

    def _rvs_v3(self, size: int):
        # # (2.14) random number generation via asymmetric correlation
        z = [ np.sqrt(1-self.delta**2) * x0 + self.delta * abs(x1) 
              for x0, x1 in multivariate_normal(cov=np.identity(2)).rvs(size=size)]  # type: ignore
        return np.array(z) if size > 1 else z[0]

    def _rvs_v4(self, size: int):
        # # (2.16) random number generation via maxima of bivariate marginals
        rho = 1 - 2 * self.delta**2
        corr = np.array([[1.0, rho], [rho, 1.0]])
        z = [ max(x) for x in multivariate_normal(cov=corr).rvs(size=size)]  # type: ignore
        return np.array(z) if size > 1 else z[0]


class SN(SN_Std, Univariate_Skew_LocScale):
    # this is (2.3), called SN(eps, w^2, alpha) in the book
    # multiple inheritance used here, this is a diamond problem
    def __init__(self, beta: float, scale=1.0, loc=0.0):
        SN_Std.__init__(self, beta=beta)
        Univariate_Skew_LocScale.__init__(self, beta=self.beta, rv_square=self.rv_square, scale=scale, loc=loc)  # type: ignore

    def mgf(self, t):
        # (2.6)
        assert isinstance(t, float), f"ERROR: t = {t} must be a float"
        return 2.0 * np.exp(t * self.loc + self.scale**2 * t**2 / 2.0) * norm.cdf(self.delta * self.scale * t)


# ---------------------------------------
# (4.11) of Azzalini and Capitanio (2014)
class ST_Std(Univariate_Skew_Std):
    def __init__(self, k, beta):
        self.k = float(k)
        rv_square = f(1, self.k)  # (4.13)
        Univariate_Skew_Std.__init__(self, beta=beta, rv_square=rv_square)
        self.b = self._b_nu()  # this is ST-specific, not the same as default

    def _pdf(self, x):
        # need this because self.pdf is ambiguous for cdf between Std and LocScale
        assert isinstance(x, float), f"ERROR: x = {x} must be a float"
        k = self.k 
        y = self.beta * x * np.sqrt((k + 1.0) / (k + x**2))
        return 2.0 * t.pdf(x, k) * t.cdf(y, k+1)

    def _pdf_square(self, x):
        return self.rv_square.pdf(x)  # type: ignore

    def _cdf(self, x):
        return quad(self._pdf, -np.inf, x)[0]
    
    def _moment(self, n):
        # my version of ST moment near (4.14)
        m_Z0 = SN_Std(self.beta)._moment(n)
        m_V = fcm_moment(-n, alpha=1.0, k=self.k)
        return m_Z0 * m_V

    def _b_nu(self):
        nu = self.k  # (4.15)
        A = nu**0.5 * gamma((nu-1)/2)
        B = np.pi**0.5 * gamma(nu/2)
        return A/B
    
    # def _mean(self):  return self.b * self.delta  # (4.16)

    def _var(self):
        return self.k / (self.k - 2) - self._mean()**2  # (4.17)

    def _skew(self):
        return self._skew_from_moments()

    def _kurtosis(self):
        return self._kurtosis_from_moments()

    def _mode(self):
        return self._mode_estimate()

    def _rvs(self, size: int, random_state=None):
        # # (2.13) random number generation
        z0 = SN_Std(self.beta)._rvs(size=size, random_state=random_state)
        v = chi(self.k).rvs(size, random_state=random_state) / self.k**0.5
        if size == 1:
            return z0 / v  # type: ignore
        else:
            return np.array(z0) / np.array(v)


class ST(ST_Std, Univariate_Skew_LocScale):
    # p.103, above (4.14)
    def __init__(self, k, beta: float, scale=1.0, loc=0.0):
        ST_Std.__init__(self, k=k, beta=beta)
        Univariate_Skew_LocScale.__init__(self, beta=self.beta, rv_square=self.rv_square, scale=scale, loc=loc)  # type: ignore


# ------------------------------------------------------------
# this is more for analytic purpose
# use gas_sn above for practical use
class GAS_SN_Std(Univariate_Skew_Std):
    def __init__(self, alpha, k, beta):
        self.alpha = float(alpha)
        self.k = float(k)
        rv_square = gas_sn_squared(self.alpha, self.k)
        Univariate_Skew_Std.__init__(self, beta=beta, rv_square=rv_square)

        self.fcm = frac_chi_mean(self.alpha, self.k)
        self.b = np.sqrt(2.0 / np.pi) * fcm_moment(-1, alpha=self.alpha, k=self.k) 
        self.c = 0.5
        
    def _quad(self, fn, a=0.0, b=np.inf):
        return quad(fn, a=a, b=b, limit=100000)[0]

    def _pdf(self, x):
        assert isinstance(x, float)
        def _kernel(s: float):
            sn_pdf = 2.0 * norm.pdf(x*s) * norm.cdf(self.beta * x*s)
            return self.fcm.pdf(s) * sn_pdf * s  # type: ignore

        return self._quad(_kernel)

    def _cdf(self, x):
        assert isinstance(x, float)

        def _kernel(s: float):
            sn_cdf = norm.cdf(x*s) - 2.0 * owens_t(x*s, self.beta)
            return self.fcm.pdf(s) * sn_cdf  # type: ignore

        return self._quad(_kernel)

    def _var(self):
        # SN's second moment is 1
        return fcm_moment(-2, alpha=self.alpha, k=self.k) - self._mean()**2

    def _moment(self, n):
        n = float(n)
        return SN_Std(self.beta)._moment(n) * fcm_moment(-n, alpha=self.alpha, k=self.k)

    def _moment_by_mellin(self, n):
        return gas_sn_moment_by_mellin(n, self.alpha, self.k, self.beta)

    def _skew(self):
        return self._skew_from_moments()
    
    def _kurtosis(self):
        return self._kurtosis_from_moments()

    def _mode(self):
        return self._mode_estimate()

    def _rvs(self, size: int, random_state=None):
        # Section 12.1
        assert isinstance(size, int) and size > 0, f"ERROR: size = {size} must be a positive integer"
        # One rng drives both draws. self.fcm is lru_cached and therefore shared, so the
        # random_state is passed per call and never set as an attribute on it.
        z0 = SN_Std(self.beta)._rvs(size=size, random_state=random_state)
        v = self.fcm.rvs(size=size, random_state=random_state)
        if size == 1:
            return z0 / v
        else:
            return np.array(z0) / np.array(v)  # type: ignore

    def _m3(self):
        def q(n): return fcm_moment(-n, self.alpha, self.k)
        delta_3 = np.sqrt(2.0 / np.pi) * self.delta * (3.0 - self.delta**2)
        m3 = delta_3 * q(3) - 3 * self._mean() * q(2) + 2 * self._mean()**3
        return m3
        
    def _skew_formula(self):
        return self._m3() / self._var()**1.5
        
    def _kurtosis_formula(self):
        def q(n): return fcm_moment(-n, self.alpha, self.k)
        kappa =  3 * ( q(4) - q(2)**2 ) - 4 * self._mean() * self._m3() + 2 * self._mean()**4
        return kappa / self._var()**2

    def _ppf(self, p):
        return gas_sn(self.alpha, self.k, self.beta).ppf(p)

    def _mellin_transform(self, s):
        if self.beta == 0: 
            return gsas_mellin_transform(s, self.alpha, self.k)
        else:
            return gas_sn_mellin_transform(s, self.alpha, self.k, self.beta)
        
    def _pdf_by_mellin(self, x):
        if x < 0:  return GAS_SN_Std(self.alpha, self.k, -self.beta)._pdf_by_mellin(-x)
        assert x > 0
        return pdf_by_mellin(x, lambda s: self._mellin_transform(s), c=self.c)

    def _lower_gamma_by_mellin(self, x):
        assert x > 0
        def _lower_gamma_mellin(s):
            return -1/s * self._mellin_transform(s+1)
        return pdf_by_mellin(x, _lower_gamma_mellin, c=self.c - 1.0)

    def _cdf_by_mellin(self, x):
        if x < 0:
            A = GAS_SN_Std(self.alpha, self.k, -self.beta)._lower_gamma_by_mellin(-x)
            return 1.0 - self._cdf(0.0) - A
        assert x > 0
        return self._lower_gamma_by_mellin(x) + self._cdf(0.0)

    def _pdf_by_log_inversion(self, x, n_u: int = 3001, lo_sd: float = 20.0, hi_sd: float = 12.0):
        """pdf of the standard GAS-SN at x (scalar or array), for low alpha and high k.

        `_pdf` integrates the library FCM pdf, which loses accuracy as alpha falls and k grows
        (on the LNN path k ~ 1 + (1/alpha - 1/2)/0.38 it is ~1e-5 down to alpha = 0.3, 100% wrong
        at 0.2, 0 at 0.15). This variant integrates the exact density of U = log V - E[log V]
        (FracChiMean.log_v_pdf) instead, on one trapezoid grid in u shared by every x:

            f(x) = int f_U(u) 2 v phi(x v) Phi(beta x v) du,   v = exp(u + E[log V])

        U is O(1) at every (alpha, k), so the grid [-lo_sd, hi_sd] x sd(U) covers it uniformly.
        Domain k > 1 or k < 0 (FracChiMean's log inversion).
        """
        u, v, fu = self._log_v_grid(n_u, lo_sd, hi_sd)
        xv = np.multiply.outer(np.asarray(x, dtype=float), v)
        integrand = 2.0 * v * np.exp(norm.logpdf(xv) + log_ndtr(self.beta * xv)) * fu
        return trapezoid(integrand, u, axis=-1)

    def _log_v_grid(self, n_u: int = 3001, lo_sd: float = 20.0, hi_sd: float = 12.0):
        # (u, v, f_U(u)) on the shared grid of the *_by_log_inversion variants, v = exp(u + E[log V])
        f = FracChiMean(self.alpha, self.k)
        m, sd = f.log_mean(), np.sqrt(f.log_var())
        u = np.linspace(-lo_sd * sd, hi_sd * sd, n_u)
        return u, np.exp(u + m), f.log_v_pdf(u)

    def _cdf_by_log_inversion(self, x):
        """cdf of the standard GAS-SN at x (scalar or array), the companion of _pdf_by_log_inversion:
        F(x) = int f_U(u) F_SN(x v) du, with the skew-normal cdf F_SN(z) = Phi(z) - 2 T(z, beta)."""
        u, v, fu = self._log_v_grid()
        xv = np.multiply.outer(np.asarray(x, dtype=float), v)
        return trapezoid((norm.cdf(xv) - 2.0 * owens_t(xv, self.beta)) * fu, u, axis=-1)

    def _squared_sf_by_log_inversion(self, q):
        """P(X^2 > q) for the standard GAS-SN, q >= 0. Z^2 ~ chi2(1) whatever beta is, so
        P(X^2 > q) = P(|Z| > sqrt(q) V) = int f_U(u) erfc(sqrt(q) v / sqrt 2) du -- independent of beta."""
        u, v, fu = self._log_v_grid()
        t = np.multiply.outer(np.sqrt(np.asarray(q, dtype=float)), v)
        return trapezoid(2.0 * norm.sf(t) * fu, u, axis=-1)

    def _squared_ppf_by_log_inversion(self, p, n_grid: int = 2001):
        """Quantiles of X^2 at probabilities p in (0, 1). log q is interpolated (monotone cubic) against
        logit P(X^2 <= q) on a log grid spanning 1e-12 to 1e8 times E[X^2]: both ends of that map are
        close to linear, so the round trip is far tighter than interpolating against the cdf itself.
        Probabilities beyond the grid's reach come back nan."""
        q = self._moment(2) * np.geomspace(1e-12, 1e8, n_grid)
        sf = self._squared_sf_by_log_inversion(q)
        cdf = 1.0 - sf
        ok = (cdf > 0) & (sf > 0)
        logit = np.log(cdf[ok]) - np.log(sf[ok])
        # strictly increasing for the interpolant: drop any point at or below the running maximum
        # (the far tails flatten into round-off)
        keep = np.concatenate([[True], logit[1:] > np.maximum.accumulate(logit)[:-1]])
        fn = PchipInterpolator(logit[keep], np.log(q[ok][keep]), extrapolate=False)
        p = np.asarray(p, dtype=float)
        return np.exp(fn(np.log(p) - np.log1p(-p)))


class GAS_SN(GAS_SN_Std, Univariate_Skew_LocScale):
    # p.103, above (4.14)
    def __init__(self, alpha, k, beta: float, scale=1.0, loc=0.0):
        GAS_SN_Std.__init__(self, alpha=alpha, k=k, beta=beta)
        Univariate_Skew_LocScale.__init__(self, beta=self.beta, rv_square=self.rv_square, scale=scale, loc=loc)  # type: ignore

    def ppf(self, p):
        return gas_sn(self.alpha, self.k, self.beta, loc=self.loc, scale=self.scale).ppf(p)

    def rvs(self, size: int, random_state=None):
       z = self._rvs(size=size, random_state=random_state)
       return z * self.scale + self.loc

    def pdf_by_log_inversion(self, x):
        # vectorized pdf for low alpha and high k; see GAS_SN_Std._pdf_by_log_inversion
        z = (np.asarray(x, dtype=float) - self.loc) / self.scale
        return self._pdf_by_log_inversion(z) / self.scale

    def cdf_by_log_inversion(self, x):
        z = (np.asarray(x, dtype=float) - self.loc) / self.scale
        return self._cdf_by_log_inversion(z)



class GAS_SN_TAIL:
    A = 1.98
    B = 1.135

    def __init__(self, alpha, k, beta):
        self.alpha = alpha
        self.k = k
        self.beta = beta

        self.delta = beta / np.sqrt(1.0 + beta**2)
        self.mu = self.A * self.delta / np.sqrt(2.0)
        self.gsas_rv = gsas(self.alpha, self.k)

        sigma_k = fcm_sigma(alpha, k)
        sigma_k_minus_1 = fcm_sigma(alpha, k - 1)
        C_alpha_k = fcm_normalization_constant(alpha, k)
        C_alpha_k_minus_1 = fcm_normalization_constant(alpha, k - 1)

        self.q = np.sqrt(1.0 + beta**2) * sigma_k / sigma_k_minus_1
        self.G = (
            np.sqrt(2.0 / np.pi)
            * C_alpha_k
            / (self.B * sigma_k_minus_1 * C_alpha_k_minus_1)
        )

    def pdf_left_tail(self, x):
        assert x < 0, f"ERROR: x = {x} must be negative"
        assert self.beta > 0, f"ERROR: beta = {self.beta} must be positive"

        unshifted = shifted_gsas_pdf(self.q * x, alpha=self.alpha, k=self.k - 1)
        shifted = shifted_gsas_pdf(
            self.q * x,
            alpha=self.alpha,
            k=self.k - 1,
            kernel_shift=self.mu,
        )
        return self.G / (self.beta * x) * (np.exp(self.mu**2 / 2.0) * shifted - unshifted)

    def gsas_pdf(self, x):
        return self.gsas_rv.pdf(x)  # type: ignore

    def pdf(self, x):
        if self.beta < 0:
            reflected = GAS_SN_TAIL(self.alpha, self.k, -self.beta)
            return reflected.pdf(-x)
        if x == 0:
            return self.gsas_pdf(0.0)
        if self.beta == 0:
            return self.gsas_pdf(x)
        if x > 0:
            return 2.0 * self.gsas_pdf(x) - self.pdf_left_tail(-x)
        return self.pdf_left_tail(x)
    
