import numpy as np
from sympy import O 
import pandas as pd
import mpmath as mp
from functools import lru_cache
from scipy.stats import rv_continuous, levy_stable, gengamma
from scipy.special import gamma, loggamma

from pandarallel import pandarallel  # type: ignore

pandarallel.initialize(verbose=1)


from .wright import wright_fn, mainardi_wright_fn, mainardi_wright_fn_slope, mp_gamma
from .wright_levy_asymp import wright_f_fn_by_levy_asymp
from .frac_gamma import frac_gamma_star
from .utils import OneSided_RVS

from .frac_gamma import frac_gamma_star


# Note: GSC is renamed to the fractional gamma distribution in 10/2025


def fg_normalization_constant(alpha, sigma, d, p):
    # Def 1, take it out so we can test more easily
    assert alpha > 0, f"ERROR: alpha={alpha} must be positive, 0 has to be handled elsewhere"
    c = abs(p)  / sigma
    return c * gamma(d*alpha/p) / gamma(d/p) if d != 0.0 else c / alpha


def fg_log_normalization_constant(alpha, log_sigma, d, p):
    assert alpha > 0, f"ERROR: alpha={alpha} must be positive, 0 has to be handled elsewhere"
    log_c = np.log(abs(p)) - log_sigma
    return log_c + loggamma(d*alpha/p) - loggamma(d/p) if d != 0.0 else log_c - np.log(alpha)


def fg_moment(n, alpha, sigma, d, p):
    n = mp.mpf(n)
    assert n != 0
    alpha = mp.mpf(alpha)
    d = mp.mpf(d)
    p = mp.mpf(p)
    if d+n == 0: n = -d + mp.mpf(0.0001)  # the formula below has a pole at d+n unfortunately
    
    if alpha != 0:
        if d != 0:
            alpha_term = mp_gamma(d/p*alpha)  / mp_gamma(d/p)  / mp_gamma((d+n)/p*alpha)
        else:
            alpha_term = 1.0 / alpha / mp_gamma(n/p*alpha)
    else:
        # alpha = 0
        alpha_term = (d+n)/d / mp_gamma(d/p) if d != 0 else n/p
    mnt = mp.power(sigma,n) * mp_gamma((d+n)/p) * alpha_term
    return float(mnt)


def fg_mellin_transform(s, alpha: float, sigma: float, d: float, p: float):
    if alpha != 0:
        if d != 0:
            alpha_term = gamma(d/p*alpha)  / gamma(d/p)  / gamma((s+d-1.0)/p*alpha)
        else:
            alpha_term = 1.0 / alpha / gamma((s-1.0)/p*alpha)
    else:
        # alpha = 0
        alpha_term = (s+d-1.0)/d / gamma(d/p) if d != 0 else (s-1.0)/p  # alpha = d = 0

    return gamma((s+d-1.0)/p) * sigma**(s-1.0) * alpha_term


# ----------------------------------------------------
# mu for RV, various implementations
def fg_q_by_f(z, dz_ratio, alpha):
    # lemma B.3 
    f = wright_f_fn_by_levy_asymp(z, alpha)
    if dz_ratio is None or z <= 0.001:
        q_nf = wright_fn(-z, -alpha, -1.0) / (-f)
    else:
        dz = z * dz_ratio
        f_dz =  wright_f_fn_by_levy_asymp(z+dz, alpha)
        q_nf = -alpha * z * (f_dz - f)/dz / (-f) + 1
    return q_nf
    
def fg_mu_by_f(x, dz_ratio, alpha, sigma, d, p):
    # lemma B.2
    # if dz_pct is None, the use Wright function, typically good for small x < 0.3
    # dz_ratio is typcally 0.0001
    z = (x/sigma)**p
    if z == 0: z = 0.001
    q_nf = fg_q_by_f(z, dz_ratio, alpha)
    return p/(2.0*alpha) * q_nf + (d/2.0 - p/(2*alpha))


def fg_mu_by_m(x, dz_ratio, alpha, sigma, d, p):
    # if dz_pct is None, the use Wright function, typically good for small x < 0.3
    # dz_ratio is typcally 0.0001
    # mainardi function is good for alpha away from 1.0, very good for alpha between 0 and 0.5
    z = (x/sigma)**p
    if z == 0: z = 0.001
    m = mainardi_wright_fn(z, alpha)
    f = m * alpha * z 
    if dz_ratio is None or z <= 0.001:
        q_nf = wright_fn(-z, -alpha, -1.0) / (-f)
    else:
        dz = z * dz_ratio
        m_dz =  mainardi_wright_fn(z+dz, alpha)
        q_nf = alpha * z * (m_dz - m)/dz / m + (alpha + 1)  # type: ignore

    return p/(2.0*alpha) * q_nf + (d/2.0 - p/(2.0*alpha))


def fg_mu_by_m_series(x, alpha, sigma, d, p):
    # mainardi function is good for alpha away from 1.0, very good for alpha between 0 and 0.5
    z = (x/sigma)**p
    m = mainardi_wright_fn(z, alpha)
    dm_dz =  mainardi_wright_fn_slope(z, alpha)
    q_nf = alpha * z * dm_dz / m + (alpha + 1)
    return p/(2.0*alpha) * q_nf + (d/2.0 -p/(2.0*alpha))


def fg_mu_at_half_alpha(x, sigma, d, p):
    z = (x/sigma)**(2*p)
    return -p/4 * z + (d+p)/2


# ----------------------------------------------------
def fg_pdf_large_x(x, alpha, sigma, d, p):
    # this formula doesn't work for small alpha
    afrac = alpha/(1.0-alpha)
    a = (1.0-alpha) * alpha**afrac
    b2 = alpha**(0.5/(1.0-alpha)) / np.sqrt((1.0-alpha) * 2.0*np.pi)
    pfrac = p/(1.0-alpha)
    c = fg_normalization_constant(alpha, sigma, d, p)
    return (b2 * c) * (x/sigma)**(d + 0.5*pfrac - 1.0) * np.exp(-a * (x/sigma)**pfrac)


def gengamma_from_gg(a, d, p):
    return gengamma(a=d/p, c=p, scale=a)


class fractional_gamma_gen(rv_continuous):

    def _pdf(self, x, alpha, sigma, d, p, *args, **kwargs):
        # handle array form
        if not isinstance(alpha, float):
            assert len(alpha) == len(x), f"ERROR: len of alpha and x: {len(alpha)} != {len(x)}"
            if len(x) == 1:  # trvial case
                return self._pdf(x[0], alpha=alpha[0], sigma=sigma[0], d=d[0], p=p[0])
            
            df = pd.DataFrame(data={'x': x, 'alpha': alpha, 'sigma': sigma, 'd': d, 'p': p})
            df['pdf'] = df.parallel_apply(lambda row: self._pdf(row['x'], alpha=row['alpha'], sigma=row['sigma'], d=row['d'], p=row['p']), axis=1)  # type: ignore
            return df['pdf'].tolist()
            # return [self._pdf(x1, df=df1, alpha=a1) for x1, df1, a1 in zip(x, df, alpha)]

        # integral form
        assert isinstance(x, float)
        alpha = float(alpha)
        sigma = float(sigma)
        d = float(d)
        p = float(p)
        assert sigma > 0

        z = (x / sigma) ** p 
        if alpha != 0.0:
            C = fg_normalization_constant(alpha, sigma, d, p)
            try:
                f_alpha = wright_f_fn_by_levy_asymp(z, alpha)
                if f_alpha == 0.0:
                    return 0.0
                x_pow_log = np.log(x / sigma) * (d-1)
                return C * np.exp( x_pow_log + np.log(f_alpha))
            except OverflowError:
                return 0.0
        else:
            # this is just generalized gamma: gengamma(a=(d+p)/p, c=p, scale=sigma)
            d = d + p
            C = abs(p) / sigma / gamma(d/p)  # odd case is IG, where d < 0 and p < 0
            try:
                x_pow_log = np.log(x / sigma) * (d-1)
                return C * np.exp(x_pow_log - z)
            except OverflowError:
                return 0.0

    def _cdf(self, x, alpha, sigma, d, p, *args, **kwargs):
        # handle array form
        if not isinstance(alpha, float):
            if len(alpha) == 1 and len(x) > 1:
                n = len(x)
                return self._cdf(x, alpha=np.repeat(alpha,n), sigma=np.repeat(sigma,n), d=np.repeat(d,n), p=np.repeat(p,n))
            assert len(alpha) == len(x), f"ERROR: len of alpha and x mismatch: {len(alpha)} != {len(x)}"
            if len(x) == 1:  # trvial case
                return self._cdf(x[0], alpha=alpha[0], sigma=sigma[0], d=d[0], p=p[0])
            
            df = pd.DataFrame(data={'x': x, 'alpha': alpha, 'sigma': sigma, 'd': d, 'p': p})
            df['cdf'] = df.parallel_apply(lambda row: self._cdf(row['x'], alpha=row['alpha'], sigma=row['sigma'], d=row['d'], p=row['p']), axis=1)  # type: ignore
            return df['cdf'].tolist()

        # ---------------------
        assert isinstance(x, float)
        # s2 = d/(2*p) + 0.5 
        # sigma2 = sigma * 2**(1/p)
        # x2 = (x/sigma2)**(2*p) 
        # return frac_gamma_inc(s2, x2, alpha)
        
        z = x / sigma
        return z**(d + p) * frac_gamma_star(d/p+1, z**p, alpha=alpha)

    def _argcheck(self, *args, **kwargs):
        # Customize the argument checking here
        alpha = args[0]
        sigma = args[1]
        d = args[2]
        p = args[3]

        return (
            alpha >= 0  # Allow alpha to be zero or positive
            and abs(d) >= 0  # for IG, d can be negative
            and p != 0
            and sigma > 0
        )

    def _munp(self, n, alpha, sigma, d, p, *args, **kwargs):
        # https://github.com/scipy/scipy/issues/13582
        # mu = self._munp(1, *goodargs)
        assert isinstance(n, int), f"ERROR: n must be int"
        if not isinstance(alpha, float):
            if len(alpha) == 1:  # trvial case
                return fg_moment(n, alpha=alpha[0], sigma=sigma[0], d=d[0], p=p[0])
            df = pd.DataFrame(data={'n': n, 'alpha': alpha, 'sigma': sigma, 'd': d, 'p': p})
            df['mnt'] = df.parallel_apply(lambda row: fg_moment(row['n'], alpha=row['alpha'], sigma=row['sigma'], d=row['d'], p=row['p']), axis=1)  # type: ignore
            return df['mnt'].tolist()

        else:
            return fg_moment(n, alpha, sigma, d, p)

    def _rvs(self, alpha, sigma, d, p, *args, **kwargs):
        size = kwargs.get('size', 1)
        alpha = float(alpha)
        sigma = float(sigma)
        d = float(d)
        p = float(p)

        m1: float = self._munp(1, alpha, sigma, d, p)  # type: ignore
        m2: float = self._munp(2, alpha, sigma, d, p)  # type: ignore
        sd: float = np.sqrt(m2 - m1**2)
        cdf_fn = lambda x: self._cdf(x, alpha=alpha, sigma=sigma, d=d, p=p)
        return OneSided_RVS(mean=m1, sd=sd, cdf_fn=cdf_fn).rvs(size)


frac_gamma = fractional_gamma_gen(name="fractional gamma", a=0, shapes="alpha, sigma, d, p")
