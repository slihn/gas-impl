
import numpy as np 
import pandas as pd

from typing import Union, Optional
from functools import lru_cache
from scipy.stats import rv_continuous, chi2, multivariate_normal
from scipy.special import gamma, gammainc, hyp1f1
from scipy.integrate import quad
from scipy.optimize import minimize

from .fcm_dist import fcm2_moment, frac_chi2_mean, fcm_moment, fcm_sigma, fcm_normalization_constant, chi2_11, chi2_11_moment
from .hyp_geo2 import frac_hyp2f1_by_alpha_k
from .utils import OneSided_RVS, moment_by_2d_int


# -----------------------------------------------------------------------
# Azzalini (2014) p.177
# fractional extension of Snedecor's F distribution

def frac_f_moment(n, alpha, d, k):
    # return chi2(df=d, scale=1/d).moment(n) * fcm2_moment(-n, alpha, k)
    return (2/d)**n * gamma(d/2 + n) / gamma(d/2) * fcm2_moment(-n, alpha, k)


def frac_f_var(alpha, d, k):
    m2 = fcm2_moment(-2, alpha, k)
    m1_sqr = fcm2_moment(-1, alpha, k)**2    
    if d == np.inf:
        return m2 - m1_sqr
    else:
        return (1.0 + 2/d) * m2 - m1_sqr


def frac_f_pdf_peak_at_zero(x, alpha, d, k):
    d2 = d / 2
    c = d2**d2 / gamma(d2)
    return c * fcm_moment(d, alpha, k) * x**(d2-1)


class frac_f_gen(rv_continuous):
    
    def assert_params(self, alpha, d, k):
        assert isinstance(alpha, float)
        assert isinstance(d, float)
        assert isinstance(k, float)
        assert alpha >= 0
        assert d > 0

    def _pdf(self, x, alpha, d, k, *args, **kwargs):
        # handle array form
        if not isinstance(alpha, float):
            assert len(alpha) == len(x), f"ERROR: len of alpha and x"
            if len(x) == 1:  # trvial case
                return self._pdf(x[0], alpha=alpha[0], d=d[0], k=k[0])
            
            df = pd.DataFrame(data={'x': x, 'alpha': alpha, 'd': d, 'k': k})
            df['pdf'] = df.parallel_apply(lambda row: self._pdf(row['x'], alpha=row['alpha'], d=row['d'], k=row['k']), axis=1)  # type: ignore
            return df['pdf'].tolist()

        # integral form
        self.assert_params(alpha, d, k)
        assert isinstance(x, float)
        if x < 0: return 0.0

        c2 = chi2(df=d)  # chi2(df=d, scale=1/d)
        f_chi2 = frac_chi2_mean(alpha=alpha, k=k)
        
        def _kernel(s: float):
            return d*s * c2.pdf(d*s*x) * f_chi2.pdf(s)  # type: ignore

        return quad(_kernel, a=0.0, b=np.inf, limit=10000)[0]

    def _cdf(self, x, alpha, d, k, *args, **kwargs):
        # handle array form
        if not isinstance(alpha, float):
            if len(alpha) == 1 and len(x) > 1:
                n = len(x)
                return self._cdf(x, alpha=np.repeat(alpha,n), d=np.repeat(d,n), k=np.repeat(k,n))
            assert len(alpha) == len(x), f"ERROR: len of alpha and x mismatch: {len(alpha)} != {len(x)}"
            if len(x) == 1:  # trvial case
                return self._cdf(x[0], alpha=alpha[0], d=d[0], k=k[0])
            
            df = pd.DataFrame(data={'x': x, 'alpha': alpha, 'd': d, 'k': k})
            df['cdf'] = df.parallel_apply(lambda row: self._cdf(row['x'], alpha=row['alpha'], d=row['d'], k=row['k']), axis=1)  # type: ignore
            return df['cdf'].tolist()

        # integral form
        self.assert_params(alpha, d, k)
        assert isinstance(x, float)
        if x < 0: return 0.0
    
        c2 = chi2(df=d)  # chi2(df=d, scale=1/d)
        f_chi2 = frac_chi2_mean(alpha=alpha, k=k)
        
        def _kernel(s: float):
            return c2.cdf(d*s*x) * f_chi2.pdf(s)  # type: ignore

        return quad(_kernel, a=0.0, b=np.inf, limit=10000)[0]

    def _argcheck(self, *args, **kwargs):
        # Customize the argument checking here
        alpha = args[0]
        d = args[1]
        k = args[2]

        return (
            alpha >= 0  # Allow alpha to be zero or positive
            and d > 0  # d is the dimension
            and k != 0  # k is the degree of freedom
        )

    def _munp(self, n, alpha, d, k, *args, **kwargs):
        assert isinstance(n, int), f"ERROR: n must be int"
        if not isinstance(alpha, float):
            if len(alpha) == 1:  # trvial case
                return frac_f_moment(n, alpha=alpha[0], d=d[0], k=k[0])
            df = pd.DataFrame(data={'n': n, 'alpha': alpha, 'd': d, 'k': k})
            df['mnt'] = df.apply(lambda row: frac_f_moment(row['n'], alpha=row['alpha'], d=row['d'], k=row['k']), axis=1)
            return df['mnt'].tolist()

        else:
            return frac_f_moment(n, alpha, d, k)

    def _rvs(self, alpha, d, k, *args, **kwargs):
        size = kwargs.get('size', 1)
        alpha = float(alpha)
        d = float(d)
        k = float(k)
        c2 = chi2(df=d, scale=1/d)
        f_chi2 = frac_chi2_mean(alpha=alpha, k=k)
        return np.array([ x/v for x,v in zip(c2.rvs(size), f_chi2.rvs(size)) ])


frac_f = frac_f_gen(name="fractional F", shapes="alpha, d, k ")


def frac_f_cdf_by_gammainc(x, alpha, d, k):
    # this is in terms of incomplete gamma function
    f_chi2 = frac_chi2_mean(alpha=alpha, k=k)
    def _kernel(s: float):
        g = gammainc(d/2, d*s*x/2)  
        return g * f_chi2.pdf(s)  # type: ignore

    return quad(_kernel, a=0.0, b=np.inf, limit=10000)[0]


def frac_f_cdf_by_hyp2f1(x, alpha, d, k, integral=False):
    # this is in terms of Frac_Hyp2f1
    sigma_k = fcm_sigma(alpha, k)
    sigma_k_d = fcm_sigma(alpha, k + d - 1)
    S = (sigma_k_d / sigma_k)**2

    C_k = fcm_normalization_constant(alpha, k)
    C_k_d = fcm_normalization_constant(alpha, k + d - 1)
    C = sigma_k**(d - 1) / np.sqrt(S) * C_k / C_k_d

    G = (d*x/2)**(d/2) / gamma(d/2 + 1)

    if integral:
        f_chi2 = frac_chi2_mean(alpha=alpha, k=k+d-1)
        def _kernel(s: float):
            M = hyp1f1(d/2, d/2+1, -d*s*x/(2*S))  
            return C * G * M * f_chi2.pdf(s) * np.sqrt(s)  # type: ignore

        return quad(_kernel, a=0.0, b=np.inf, limit=10000)[0]
    else:
        f_hyp2f1 = frac_hyp2f1_by_alpha_k(alpha, k+d-1, a=d/2, c=d/2+1)
        return C * G * f_hyp2f1.scaled_integral(-d*x/(2*S))


# -----------------------------------------------------------------------
class Frac_F_Std_Adp_2D:
    def __init__(self, alpha, k, rho=0.0):
        self.alpha = np.array(alpha)  # type: ignore
        self.k = np.array(k)  # type: ignore
        self.rho = rho
        self.n = 2
        assert len(self.alpha) == self.n
        assert len(self.k) == self.n

        self.fcm2_list = [frac_chi2_mean(alpha=self.alpha[i], k=self.k[i]) for i in range(self.n)]
 
    def _fcm2_pdf_prod(self, s) -> float:
        fcm2_pdfs = np.array([self.fcm2_list[i].pdf(s[i]) for i in range(self.n)])   # type: ignore
        return  np.prod(fcm2_pdfs)

    def _pdf1(self, x, use_mp=True):
        assert isinstance(x, float)
        # this is only for 2D
        return moment_by_2d_int(self, p1=0, p2=0, integrand_x=x, one_sided=True, use_mp=use_mp)

    def _cdf1(self, x, use_mp=True):
        assert isinstance(x, float)
        # this is only for 2D
        return moment_by_2d_int(self, p1=0, p2=0, integrand_x=x, use_cdf_integrand=True, one_sided=True, use_mp=use_mp)
        
    def _pdf1_integrand(self, x: np.ndarray, s: np.ndarray):
        if s[0] <= 0 or s[1] <= 0: return 0.0
        c2 = chi2_11(s[0], s[1], rho=self.rho)
        return  self.n * self._fcm2_pdf_prod(s) * c2.pdf(self.n * x)  # type: ignore

    def _cdf1_integrand(self, x: np.ndarray, s: np.ndarray):
        if s[0] <= 0 or s[1] <= 0: return 0.0
        c2 = chi2_11(s[0], s[1], rho=self.rho)
        return  self._fcm2_pdf_prod(s) * c2.cdf(self.n * x)  # type: ignore

    @property
    def ff_list(self):  return [frac_f(alpha=self.alpha[i], d=1.0, k=self.k[i]) for i in range(self.n)]
    
    def pdf(self, x):
        if isinstance(x, np.ndarray):
            return np.array([self._pdf1(xi) for xi in x])
        return self._pdf1(x)
    
    def cdf(self, x):
        if isinstance(x, np.ndarray):
            return np.array([self._cdf1(xi) for xi in x])
        return self._cdf1(x)

    def _moment_integrand(self, x: np.ndarray, s: np.ndarray):
        # x is m
        if s[0] <= 0 or s[1] <= 0: return 0.0
        # c2 = chi2_11(s[0], s[1], rho=self.rho)
        # mx = c2.moment(x)
        mx = chi2_11_moment(x, s[0], s[1], rho=self.rho)  # prefer my own implementation, more certain if there is an issue
        return  self.n**(-x) * self._fcm2_pdf_prod(s) * mx
    
    def moment(self, m, use_mp=True):
        # TODO this is very complicated integral, is it really worth the effort?
        return moment_by_2d_int(self, p1=0, p2=0, integrand_x=float(m), use_moment_integrand=True, one_sided=True, use_mp=use_mp)
    
    def mean(self):  return self.moment(1)

    def var(self):  return self.moment(2) - self.mean()**2
    
    def std(self):  return self.var()**0.5

    def rvs(self, size) -> np.ndarray:
        rho = float(self.rho)
        corr = np.array([[1.0, rho], [rho, 1.0]])
        x: np.ndarray = multivariate_normal(cov=corr).rvs(size=size)  # type: ignore
        v1 = self.fcm2_list[0].rvs(size)
        v2 = self.fcm2_list[1].rvs(size)
        def Q(i):  return (x[i,0]**2/v1[i] + x[i,1]**2/v2[i]) / self.n  # Q = 1/d * (X1^2/v1 + X2^2/v2)
        return np.array([Q(i) for i in range(size)])

    def find_median(self, tol=0.001):
        def _fn(x):  return (self.cdf(x) - 0.5)**2
        x = minimize(_fn, 1.0, tol=tol).x[0]
        return float(x)

    def _conv_pdf_via_frac_f_2d(self, x, a=0.0, b=1.0, use_mp=True):
        # this is slow, for proof of concept
        assert x >= 0
        ff_list = self.ff_list
        z = self.n*x

        def _conv(t):  return ff_list[0].pdf(z*t) * ff_list[1].pdf(z*(1-t))  # type: ignore

        def _conv_u(u):  # t = u**2
            return _conv(u**2) * 2*u # type: ignore

        def _conv_u2(u):  # t = 1.0-u**2
            return _conv(1.0-u**2) * (2*u) # type: ignore

        if a == 0.0 and b == 1.0:  
            if not use_mp: # very slow
                return quad(_conv, a, b, limit=100000)[0] * self.n * z
            else:
                dt = 1.0 / 8
                df = pd.DataFrame({'t': [t*dt for t in range(int(1/dt))]})
                df['pdf'] = df['t'].parallel_apply(lambda t: self._conv_pdf_via_frac_f_2d(x, a=t, b=t + dt, use_mp=False))
                return df.pdf.sum()

        # this is much faster, since it deals with the singularity at t=0 and t=1
        if not np.allclose(b, 1.0):  # type: ignore
            return quad(_conv_u, a, b, limit=100000)[0] * self.n * z
        else:
            # handle b = 1
            a2 = np.sqrt(1-a**2)
            b2 = np.sqrt(1-b**2)
            return quad(_conv_u2, b2, a2, limit=100000)[0] * self.n * z


# -----------------------------------------------------------------------
# the default ppf() is slow, the following has a warmup time of 50 seconds, then it is very fast
# but it is not as accurate as the default ppf()
# the main use case is to generate QQ-plot
class FracF_PPF:
    def __init__(self, alpha, d: Union[float, int], k, rho: Optional[float]=None, 
                 mean_override: Optional[float]=None, sd_override: Optional[float]=None,
                 RV_override = None,
                 delta_cdf: float = 1e-6, interp_size = 2001):
        self.alpha = alpha
        self.d = float(d)
        self.k = k
        self.rho = rho
        self.mean_override = mean_override
        self.sd_override = sd_override
        self.use_mp = True  # _get_rv can change it
        if RV_override is not None:
            self.rv = RV_override
        else:
            self.rv = self._get_rv()

        self.delta_cdf = delta_cdf
        self.interp_size = interp_size

        self.cdf_fn = lambda x: self.rv.cdf(x)
        
        mean = self.mean_override if self.mean_override is not None else self.rv.mean()
        sd = self.sd_override if self.sd_override is not None else self.rv.std()

        if RV_override is not None:
            self.RVS = RV_override
        else:
            self.RVS = OneSided_RVS(mean=mean, sd=sd, cdf_fn=self.cdf_fn,
                                delta_cdf=delta_cdf, interp_size=interp_size, use_mp=self.use_mp)
    
        self.observed_data = np.array([])

    def moment(self, n):
        return self.rv.moment(n)

    def _get_rv(self):
        if self.rho is None:
            # elliptical case, no correlation
            return frac_f(alpha=self.alpha, d=self.d, k=self.k)

        # otherwise, adaptive case, alpha and k are 2D, rho is scalar
        assert self.d == 2.0
        assert abs(self.rho) <= 1.0
        assert len(self.alpha) == 2 and len(self.k) == 2
        self.use_mp = False
        return Frac_F_Std_Adp_2D(alpha=self.alpha, k=self.k, rho=self.rho)

    def ppf(self, x):
        return self.RVS.ppf(np.array(x))
    
    def cdf(self, x):
        return self.RVS.cdf(np.array(x))

    def pdf(self, x):
        return self.rv.pdf(x)  # type: ignore

    # this is for stanardized PP-plot and QQ-plot
    def set_observed_data(self, data):
        self.observed_data = np.sort(np.array(data))

    @property
    def observed_cdf(self):
        n = len(self.observed_data)
        return np.arange(1, n + 1) / (n + 1)

    def analyze_quantiles(self):
        theoretical = self.ppf(self.observed_cdf)
        return self.observed_data, theoretical
    
    def analyze_cdf(self):
        theoretical_cdf = self.cdf(self.observed_data)
        return self.observed_cdf, theoretical_cdf

# -----------------------------------------------------------------------


