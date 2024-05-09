import numpy as np 
import pandas as pd
import mpmath as mp
from functools import lru_cache
from scipy.stats import rv_continuous, levy_stable, gengamma
from scipy.special import gamma
from scipy.special import sici 
from scipy.integrate import quad, quadrature
from pandarallel import pandarallel  # type: ignore

pandarallel.initialize(verbose=1)

from .wright import wright_fn, mainardi_wright_fn, mainardi_wright_fn_slope, mp_gamma

# pyright: reportGeneralTypeIssues=false

# --------------------------------------------------------------------------------
def stable_count_pdf_small_x(x, alpha):
    # this is only good for large alpha
    assert alpha > 0 and alpha < 1.0
    q = alpha / gamma(1.0/alpha+1) / gamma(1.0-alpha)
    return q * x**alpha


def stable_count_pdf_wright(x, alpha, max_n=10):
    # this is a replacement of small x asymptotic, especially for small alpha
    assert alpha > 0 and alpha < 1.0
    q = 1.0 / gamma(1.0/alpha+1)
    return q * wright_fn(-x**alpha, lam=-alpha, mu=0.0, max_n=max_n)


def stable_count_pdf_large_x(x, alpha):
    # this formula doesn't work for small alpha
    afrac = alpha/(1.0-alpha)
    a = (1.0-alpha) * alpha**afrac
    b = alpha**(0.5/(1.0-alpha)) / gamma(1.0/alpha+1) / np.sqrt((1.0-alpha) * 2.0*np.pi)
    return b * x**(0.5*afrac) * np.exp(-a * x**afrac)


def stable_count_moment(n, alpha):
    return gamma((n+1)/alpha) / gamma(n+1) / gamma(1/alpha)


def levy_stable_one_sided(alpha):
    assert 0 < alpha <= 1.0
    scale = np.power(np.cos(alpha * np.pi / 2.0), 1.0/alpha)
    return levy_stable(alpha, beta=1, loc=0, scale=scale)


# https://github.com/scipy/scipy/blob/v1.8.0/scipy/stats/_continuous_distns.py
# above URL contains many examples how real distributions are implemented


class stable_count_gen(rv_continuous):

    @staticmethod 
    def q(alpha):
        q1 = np.cos(alpha * np.pi/2)
        q2 = np.sin(-alpha * np.pi/2)
        return (q1, q2)
    
    @staticmethod
    @lru_cache(maxsize=100)
    def rv_stable_one_sided(alpha):
        return levy_stable_one_sided(alpha)

    def _munp(self, n, alpha, *args, **kwargs):
        # https://github.com/scipy/scipy/issues/13582
        return stable_count_moment(n, alpha)

    def _pdf(self, x, alpha, *args, **kwargs):
        if isinstance(alpha, float):
            assert x >= 0
            if x == 0.0: return 0.0  # N(0) = 0
            rvl = self.rv_stable_one_sided(alpha)
            return rvl.pdf(1.0/x) / x / gamma(1/alpha+1)
        else:
            assert len(alpha) == len(x), f"ERROR: len of alpha and x"
            return [self._pdf(x1, alpha=a1) for x1, a1 in zip(x, alpha)]

    def ccdf_int(self, x, alpha, method="quad"):
        # Note: maybe we don't need this here!
        # broadcast x and alpha
        q1, q2 = self.q(alpha)
        c = 2.0 / np.pi / gamma(1.0/alpha + 1)
        fn = lambda t: np.exp(-q1 * np.power(t, alpha)) * np.sin(-q2 * np.power(t, alpha)) * sici(t/x)[0] * c
        if method == "quad":
            rs = quad(fn, a=0, b=np.inf, limit=1000)
            return rs[0]
        if method == "gaussian":
            rs = quadrature(fn, a=0, b=1000+x, maxiter=1000)
            return rs[0]
        raise Exception("ERROR: Unknown integration method")


stable_count = stable_count_gen(name="stable count", a=0, shapes="alpha")


def wright_f_fn_by_sc(x, alpha: float):
    nu = x**(1/alpha)
    return stable_count(alpha).pdf(nu) * gamma(1.0/alpha + 1)


###################################################################################
#
# generalized stable count distribution: implemented through the stable count above
#
def gsc_normalization_constant(alpha, sigma, d, p):
    # Def 1, take it out so we can test more easily
    return abs(p)  / sigma * gamma(d*alpha/p) / gamma(d/p) if d != 0.0 else abs(p) / alpha / sigma


def gsc_moment(n, alpha, sigma, d, p):
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


# ----------------------------------------------------
# mu for RV, various implementations
def gsc_q_by_f(z, dz_ratio, alpha):
    # lemma B.3 
    f = wright_f_fn_by_sc(z, alpha)
    if dz_ratio is None or z <= 0.001:
        q_nf = wright_fn(-z, -alpha, -1.0) / (-f)
    else:
        dz = z * dz_ratio
        f_dz =  wright_f_fn_by_sc(z+dz, alpha)
        q_nf = -alpha * z * (f_dz - f)/dz / (-f) + 1
    return q_nf
    
def gsc_mu_by_f(x, dz_ratio, alpha, sigma, d, p):
    # lemma B.2
    # if dz_pct is None, the use Wright function, typically good for small x < 0.3
    # dz_ratio is typcally 0.0001
    z = (x/sigma)**p
    if z == 0: z = 0.001
    q_nf = gsc_q_by_f(z, dz_ratio, alpha)
    return p/(2.0*alpha) * q_nf + (d/2.0 - p/(2*alpha))


def gsc_mu_by_m(x, dz_ratio, alpha, sigma, d, p):
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
        q_nf = alpha * z * (m_dz - m)/dz / m + (alpha + 1)

    return p/(2.0*alpha) * q_nf + (d/2.0 - p/(2.0*alpha))


def gsc_mu_by_m_series(x, alpha, sigma, d, p):
    # mainardi function is good for alpha away from 1.0, very good for alpha between 0 and 0.5
    z = (x/sigma)**p
    m = mainardi_wright_fn(z, alpha)
    dm_dz =  mainardi_wright_fn_slope(z, alpha)
    q_nf = alpha * z * dm_dz / m + (alpha + 1)
    return p/(2.0*alpha) * q_nf + (d/2.0 -p/(2.0*alpha))


def gsc_mu_at_half_alpha(x, sigma, d, p):
    z = (x/sigma)**(2*p)
    return -p/4 * z + (d+p)/2


# ----------------------------------------------------
def gsc_pdf_large_x(x, alpha, sigma, d, p):
    # this formula doesn't work for small alpha
    afrac = alpha/(1.0-alpha)
    a = (1.0-alpha) * alpha**afrac
    b2 = alpha**(0.5/(1.0-alpha)) / np.sqrt((1.0-alpha) * 2.0*np.pi)
    pfrac = p/(1.0-alpha)
    c = gsc_normalization_constant(alpha, sigma, d, p)
    return (b2 * c) * (x/sigma)**(d + 0.5*pfrac - 1.0) * np.exp(-a * (x/sigma)**pfrac)


def gengamma_from_gg(a, d, p):
    return gengamma(a=d/p, c=p, scale=a)


class generalized_stable_count_gen(rv_continuous):

    @lru_cache(maxsize=100)
    def stable_count(self, alpha):
        return stable_count(alpha)

    @staticmethod
    @lru_cache(maxsize=100)
    def rv_stable_one_sided(alpha):
        return levy_stable_one_sided(alpha)

    def _pdf(self, x, alpha, sigma, d, p, *args, **kwargs):
        # handle array form
        if not isinstance(alpha, float):
            assert len(alpha) == len(x), f"ERROR: len of alpha and x"
            if len(x) == 1:  # trvial case
                return self._pdf(x[0], alpha=alpha[0], sigma=sigma[0], d=d[0], p=p[0])
            
            df = pd.DataFrame(data={'x': x, 'alpha': alpha, 'sigma': sigma, 'd': d, 'p': p})
            df['pdf'] = df.parallel_apply(lambda row: self._pdf(row['x'], alpha=row['alpha'], sigma=row['sigma'], d=row['d'], p=row['p']), axis=1)
            return df['pdf'].tolist()
            # return [self._pdf(x1, df=df1, alpha=a1) for x1, df1, a1 in zip(x, df, alpha)]

        # integral form
        assert isinstance(x, float)
        alpha = float(alpha)
        sigma = float(sigma)
        d = float(d)
        p = float(p)
        assert sigma > 0

        x_pow = (x / sigma) ** (d-1)
        if alpha != 0.0:
            C = gsc_normalization_constant(alpha, sigma, d, p)

            x_pow = (x / sigma) ** (d-1)
            rv_sc = self.stable_count(alpha)
            nu = (x / sigma) ** (p/alpha) 
            sc = gamma(1.0/alpha+1) * rv_sc.pdf(nu)
            return C * x_pow * sc
        else:
            # this is just generalized gamma: gengamma(a=(d+p)/p, c=p, scale=sigma)
            d = d + p
            C = abs(p) / sigma / gamma(d/p)  # odd case is IG, where d < 0 and p < 0
            x_pow = (x / sigma) ** (d-1)
            z = (x / sigma) ** p 
            return C * x_pow * np.exp(-z) 

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
                return gsc_moment(n, alpha=alpha[0], sigma=sigma[0], d=d[0], p=p[0])
            df = pd.DataFrame(data={'n': n, 'alpha': alpha, 'sigma': sigma, 'd': d, 'p': p})
            df['mnt'] = df.parallel_apply(lambda row: gsc_moment(row['n'], alpha=row['alpha'], sigma=row['sigma'], d=row['d'], p=row['p']), axis=1)
            return df['mnt'].tolist()

        else:
            return gsc_moment(n, alpha, sigma, d, p)


gen_stable_count = generalized_stable_count_gen(name="generalized stable count", a=0, shapes="alpha, sigma, d, p")

# -----------------------------------
# constructors of known distributions
# -----------------------------------

class stable_vol_gen(rv_continuous):

    def _munp(self, n, alpha):
        # https://github.com/scipy/scipy/issues/13582
        # mu = self._munp(1, *goodargs)
        if n != -1.0:
            return gamma((n+1)/alpha) / gamma((n+1)/2) / gamma(1/alpha) * np.sqrt(np.pi) * np.power(2.0, -n/2)
        else:
            return 1.0 / 2 / gamma(1/alpha+1) * np.sqrt(np.pi) * np.power(2.0, -n/2)

    def _pdf(self, x, alpha):
        if isinstance(alpha, float):
            assert 0 < alpha <= 2.0
            rv_sc = stable_count(alpha/2.0)
            c = np.sqrt(np.pi*2) * gamma(2/alpha+1) / gamma(1/alpha+1)
            return rv_sc.pdf(2.0 * x*x) * c
        else:
            assert len(alpha) == len(x), f"ERROR: len of alpha and x"
            return [self._pdf(x1, alpha=a1) for x1, a1 in zip(x, alpha)]


stable_vol = stable_vol_gen(name="stable vol", a=0, shapes="alpha")


def sv_mu_by_f(x, dz_ratio, alpha):
    return gsc_mu_by_f(x, dz_ratio, alpha=alpha/2, sigma=1.0/np.sqrt(2), d=1.0, p=alpha)
