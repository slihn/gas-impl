
import numpy as np 
import pandas as pd
import mpmath as mp
import multiprocessing

from typing import Union, Optional, Callable, List
from datetime import datetime
from numpy import polyfit, poly1d  # type: ignore

from scipy.stats import skew, kurtosis, uniform
from scipy.integrate import quad, dblquad
from scipy.linalg import eigh, inv
from scipy.interpolate import interp1d

from .int_utils import PDF_2D_Integration, PDF_Marginal_Integration


def calc_elasticity(fn, x, d_log_x=0.001, debug=False):
    if isinstance(x, (list, np.ndarray, pd.Series)):
        rs = np.array([calc_elasticity(fn, xi, d_log_x=d_log_x) for xi in x])  # type: ignore
        return make_list_type(rs, x)

    x1 = float(x)
    x2 = x1 * np.exp(d_log_x) if x1 != 0 else x1 + d_log_x  # x + dx, exceptional usage for x = 0
    assert isinstance(x1, float) and isinstance(x2, float)
    m1 = fn(x1)
    m2 = fn(x2)
    if isinstance(m1, list):
        m1 = m1[0]
    if isinstance(m2, list):
        m2 = m2[0]
    elasticity = np.log(m2 / m1) / d_log_x
    if debug:
        print(f"x1: {x1}, x2: {x2}, m1: {m1}, m2: {m2}, elasticity: {elasticity}")
    return elasticity


def calc_elasticity_mp(fn, x, d_log_x=mp.mpf(0.001), debug=False):
    if isinstance(x, (list, np.ndarray, pd.Series)):
        rs = np.array([calc_elasticity_mp(fn, xi, d_log_x=d_log_x) for xi in x])  # type: ignore
        return make_list_type(rs, x)

    x1 = mp.mpf(x)
    x2 = x1 * mp.exp(d_log_x) if x1 != 0 else x1 + d_log_x  # x + dx, exceptional usage for x = 0
    assert isinstance(x1, mp.mpf) and isinstance(x2, mp.mpf)
    assert d_log_x > 0, f"ERROR: d_log_x={float(d_log_x)} must be positive"
    m1 = fn(x1)
    m2 = fn(x2)
    if isinstance(m1, list):
        m1 = m1[0]
    if isinstance(m2, list):
        m2 = m2[0]
    assert m1 != 0 and m2 != 0, f"ERROR: m1={float(m1)} at x1={float(x1)}, m2={float(m2)} at x2={float(x2)} must be non-zero"
    elasticity = mp.log(m2 / m1) / d_log_x
    if debug:
        print(f"x1: {float(x1)}, x2: {float(x2)}, m1: {float(m1)}, m2: {float(m2)}, elasticity: {float(elasticity)}")
    return elasticity


def make_list_type(rs: np.ndarray, x: Union[List, np.ndarray, pd.Series]):
    if isinstance(x, pd.Series):
        return pd.Series(rs, index=x.index)
    if isinstance(x, list):
        return list(rs)
    return rs


def calc_stats_from_moments(m):
    # m can be a list of moments from 0 to 4, or a pd.Series indexed by n-th moment (0 to 4)
    var = m[2] - m[1]**2
    cm3 = m[3] -3*m[1]*m[2] + 2*m[1]**3
    cm4 = m[4] -4*m[1]*m[3] + 6*m[1]**2 *m[2] -3*m[1]**4

    return {
        'total': m[0],
        'mean': m[1], 
        'var': var, 
        'skew': cm3 / np.sqrt(var**3) if var > 0 else np.nan, 
        'kurtosis': (cm4 / var**2) - 3.0,  # this is excess kurtosis (normal kurtosis is 0.0)
    }


def calc_mvsk_stats(z):
    assert isinstance(z, np.ndarray) or isinstance(z, pd.Series)
    return [ z.mean(), z.var(), skew(z), kurtosis(z) ]


def is_number(s):
    """ Returns True if string is a number. """
    try:
        float(s)
        return True
    except ValueError:
        return False

def quadratic_form(Z: np.ndarray, cov: np.ndarray, loc: Optional[np.ndarray] = None) -> np.ndarray:
    # calculate Z.T @ cov^{-1} @ Z
    if loc is not None:
        assert isinstance(loc, np.ndarray)
        Z = np.array([ z - loc for z in Z])
    cov_inv = inv(cov)
    d = cov.shape[0]
    Q = 1.0/d * np.array([(z.T @ cov_inv @ z) for z in Z])  # type: ignore
    return np.sort(Q)


class OneSided_RVS_Base:
    # construct a one-sided distribution from a given RVS
    def __init__(self, interp_size):
        self.interp_size = interp_size
        self.largest_x = np.nan
        self.smallest_x = np.nan  # will be determined later
        self.max_cdf: float = np.nan  # will be determined later
        self.min_cdf: float = np.nan  # will be determined later

        self.ppf_interp_fn: Optional[Callable] = None
        self.cdf_interp_fn: Optional[Callable] = None

    def ppf_interp(self, size, debug=False):  
        # this function should build interp1d objects and all the side effects
        pass 

    def rvs(self, size):
        # size from rv_generic._rvs() is a tuple, like (1000,)
        if self.ppf_interp_fn is not None:
            ppf_interp = self.ppf_interp_fn
        else:
            ppf_interp = self.ppf_interp(np.sum(size))

        assert self.max_cdf > 0.0
        assert ppf_interp is not None
        # print(f"max_cdf: {self.max_cdf}, largest_x: {self.largest_x}")

        def ppf(p):
            assert np.all(0 <= p) and np.all(p <= 1)
            p = np.clip(p, a_min=self.min_cdf, a_max=self.max_cdf)  # type: ignore
            return ppf_interp(p)

        return ppf(uniform.rvs(size=size))

    def ppf(self, p):
        # this function should be called from rv_generic._ppf()
        if self.ppf_interp_fn is None:
            self.ppf_interp_fn = self.ppf_interp(self.interp_size)
        assert self.max_cdf > 0.0
        assert self.ppf_interp_fn is not None

        assert np.all(0 <= p) and np.all(p <= 1)
        p = np.clip(p, a_min=self.min_cdf, a_max=self.max_cdf)  # type: ignore
        y = self.ppf_interp_fn(p)
        return float(y) if isinstance(p, float) else y

    def cdf(self, x):
        if self.cdf_interp_fn is None:
            self.ppf_interp(self.interp_size)
        
        assert self.cdf_interp_fn is not None
        assert np.all(0 <= x) 
        x = np.clip(x, a_min=self.smallest_x, a_max=self.largest_x)  # type: ignore
        y = self.cdf_interp_fn(x)
        return float(y) if isinstance(x, float) else y

    def moment(self, n):
        def _kernel(x):  return self.ppf(x)**n 
        return quad(_kernel, self.min_cdf, self.max_cdf, limit=100000, epsrel=0.001)[0]

    def mean(self):  return self.moment(1)

    def var(self):
        m1 = self.moment(1)
        m2 = self.moment(2)
        return m2 - m1**2

    def std(self):
        return np.sqrt(self.var())


class OneSided_RVS(OneSided_RVS_Base):
    # this is for one-sided distribution, like GSC, FCM, who's PDF is predictable
    # this utility assumes no loc involved, minimum support is x = 0
    # it should be called from inside rv._rvs() method
    def __init__(self, mean: float, sd: float, cdf_fn, delta_cdf: float = 1e-5, interp_size = 1001, use_mp=True):
        super().__init__(interp_size)
        self._mean = mean
        self._sd = sd
        self.cdf_fn = cdf_fn
        self.delta_cdf = delta_cdf
        self.use_mp = use_mp
        self.interp_min_size = 101

    def find_largest_x(self):
        x = self._mean + self._sd  # start from mean + 1*sd 
        while x < self._mean + 100 * self._sd:
            p = self.cdf_fn(x)
            if 1.0 - p < self.delta_cdf: break
            x += self._sd
        return x

    def ppf_interp(self, size: int, debug=False):
        if debug:  print(f"setting up interp for size: {size}, time: {datetime.now().isoformat()}")
        intervals = self.interp_size if size >= self.interp_size else min([size, self.interp_min_size])
        max_x = self.find_largest_x()
        x1 = np.linspace(self._mean, max_x, intervals)
        sz = self.interp_min_size
        x2 = np.linspace(0.11, self._mean, sz)
        x3 = np.linspace(0.011, self._mean/10, sz)
        x4 = np.linspace(0.0011, self._mean/100, sz)
        df = pd.DataFrame({'x': np.concatenate((x1, x2, x3, x4)) }).round(3).drop_duplicates()
        df = df.sort_values(by=['x'])  # type: ignore
        if self.use_mp:
            df['cdf'] = df.x.parallel_apply( lambda x: self.cdf_fn(x) )  # Important: it uses parallel_apply, and calls _cdf()
        else:
            df['cdf'] = df.x.apply( lambda x: self.cdf_fn(x) )  # Adp can't use parallel_apply

        self.max_cdf = df.cdf.max()
        self.min_cdf = df.cdf.min()
        self.largest_x = df.x.max()  # this is overwritten but basically the same number
        self.smallest_x = df.x.min() 
        fn = interp1d(df.cdf, df.x, kind='cubic')
        if size >= self.interp_size:
            self.ppf_interp_fn = fn  # store the good one for reuse
            self.cdf_interp_fn = interp1d(df.x, df.cdf, kind='cubic')
        if debug:  print(f"finished interp for size: {size}, time: {datetime.now().isoformat()}")
        return fn
    


class OneSided_From_RVS(OneSided_RVS_Base):
    # construct a one-sided distribution from a given RVS
    # the use case is for adaptive distribution, where close form of its quadratic form is not known yet
    # lcc := log ccdf = log(1 - cdf)
    def __init__(self, samples):
        self.samples = np.sort(samples)
        super().__init__(interp_size=len(self.samples))
        self._mean = np.mean(self.samples)
        self._lcc_poly_range = -6.0
        self._lcc_tail_start = -5.0
        self._lcc_tail_end = -8.0

        self.df = pd.DataFrame()  # for debugging, assigned from ppf_interp() call
        self.cdf_exponent = np.nan
        self.cdf_tail_multiplier = np.nan 
        self.ppf_exponent = np.nan  
        self.ppf_tail_multiplier = np.nan

        self.ppf_interp(self.interp_size)  # set up the interp1d objects
    
    def _setup_df(self):
        # size input is useless, just for compatibility
        x0 = np.array([0.0])
        df = pd.DataFrame(data={'x': np.concatenate([x0, self.samples])})
        df['dx'] = df.x.diff()

        df.index.name = 'n'  # zero-based index
        df = df.reset_index()

        size = len(df)
        assert size == df.n.max() + 1
        df['cdf'] = df['n'] / (size + 1.0)
        df['lcc'] = np.log(1.0 - df.cdf)  # type: ignore
        self.df = df

        self.max_cdf = df.cdf.max()
        self.min_cdf = df.cdf.min()
        self.largest_x = df.x.max()  
        self.smallest_x = df.x.min()
        self.cdf_exponent = self._get_lcc_exponent(cdf=True)
        self.ppf_exponent = self._get_lcc_exponent(cdf=False)

    def tail_lcc(self, x):
        x_valid = np.where(x > 0, x, np.nan) 
        return self.cdf_tail_multiplier + self.cdf_exponent * np.log(x_valid)

    def tail_log_ppf(self, lcc):
        return self.ppf_tail_multiplier + self.ppf_exponent * lcc  # log(x)

    def ppf_interp(self, size, debug=False):
        self._setup_df()
        x2lcc_poly = self._get_poly_fit(cdf=True)  # x to llc
        lcc2x_poly = self._get_poly_fit(cdf=False)  # llc to x

        x_cut = self.df.query("lcc >= @self._lcc_tail_start").x.max()
        lcc_cut = x2lcc_poly(x_cut)
        assert np.allclose(lcc_cut, self._lcc_tail_start, rtol=0.1), f"ERROR: lcc_cut {lcc_cut} is too far from _lcc_tail_start {self._lcc_tail_start}"  # type: ignore
        self.cdf_tail_multiplier = lcc_cut - self.cdf_exponent * np.log(x_cut)

        def _cdf_fn(x):
            lcc = np.where(x <= x_cut, x2lcc_poly(x), self.tail_lcc(x))
            return 1.0 - np.exp(lcc)
        
        self.cdf_interp_fn = _cdf_fn
        # -----------------------------------------------------------

        ppf_cut = np.log(lcc2x_poly(lcc_cut))
        self.ppf_tail_multiplier = ppf_cut - self.ppf_exponent * lcc_cut
        if debug:
            print(f"x_cut: {x_cut}, lcc_cut: {lcc_cut}, ppf_cut: {ppf_cut}")
            print(f"cdf_exponent: {self.cdf_exponent}, multiplier: {self.cdf_tail_multiplier}")
            print(f"ppf_exponent: {self.ppf_exponent}, multiplier: {self.ppf_tail_multiplier}")

        def _ppf_fn(p):
            lcc = np.log(1 - p)
            return np.where(lcc >= -5.0, lcc2x_poly(lcc), np.exp(self.tail_log_ppf(lcc)))

        self.ppf_interp_fn = _ppf_fn
        return self.ppf_interp_fn

    def _get_poly_fit(self, cdf=True, remove_intercept=True):
        df = self.df.query("lcc >= @self._lcc_poly_range").copy()
        if cdf == True:
            coefficients = polyfit(df.x, df.lcc, 10)  
        else:
            coefficients = polyfit(df.lcc, df.x, 10)
        if remove_intercept:
            coefficients[-1] = 0.0
        # I actually don't want to the intercept since I know that x = 0, lcc = 0
        return poly1d(coefficients)

    def _get_lcc_exponent(self, cdf=True):
        a_lcc = self._lcc_tail_start 
        b_lcc = self._lcc_tail_end
        df2 = self.df.query("lcc <= @a_lcc and lcc >= @b_lcc").copy()
        df2['lx'] = np.log(df2.x)  # type: ignore
        if cdf == True:
            coefficients = polyfit(df2.lx, df2.lcc, 1)
        else:
            coefficients = polyfit(df2.lcc, df2.lx, 1)
        return coefficients[0] 


# ----------------------------------------------------------
def nearestPD(A):
    B = (A + A.T) / 2
    U, s, V = np.linalg.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))  # type: ignore
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(eigh(A3)[0]))  # type: ignore
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    try:
        np.linalg.cholesky(B)  # type: ignore
        return True
    except np.linalg.LinAlgError:  # type: ignore
        return False


# ----------------------------------------------------------
def dblquad_full_2d(pdf, epsabs=1e-3, epsrel=1e-3):
    # this is quite slow
    return dblquad(
        pdf, 
        -np.inf, np.inf, -np.inf, np.inf,
        epsabs=epsabs,  # Allow larger absolute error
        epsrel=epsrel)[0]


def moment_by_2d_int(dist, p1, p2, integrand_x=None, 
                     use_cdf_integrand=False, use_moment_integrand=False, 
                     one_sided=False,
                     use_mp=True, epsabs=1e-3, epsrel=1e-3):
    # default is use_mp=True: use multiprocessing
    # integrand_x: this is to twist the class to work for Adaptive PDF integral, set p1=p2=0 in such case

    integrator = PDF_2D_Integration(dist, p1, p2, integrand_x=integrand_x, 
                                    use_cdf_integrand=use_cdf_integrand,
                                    use_moment_integrand=use_moment_integrand,
                                    one_sided=one_sided,
                                    epsabs=epsabs, epsrel=epsrel)
    if not use_mp:  return integrator.calc()

    processes = [
        multiprocessing.Process(target=integrator.calc_slice, args=(i,)) 
        for i in range(1, integrator.slices+1)
    ]

    for p in processes:  p.start()
    for p in processes:  p.join()
    rs = []
    while not integrator.queue.empty():
        rs.append(integrator.queue.get())
    return sum(rs)


def marginal_1d_pdf_by_int(dist, x, n, use_mp=True, epsabs=1e-3, epsrel=1e-3):
    # default is use_mp=True: use multiprocessing

    integrator = PDF_Marginal_Integration(dist, x, n, epsabs, epsrel)
    if not use_mp:  return integrator.calc()

    processes = [
        multiprocessing.Process(target=integrator.calc_slice, args=(i,)) 
        for i in range(1, integrator.slices+1)
    ]

    for p in processes:  p.start()
    for p in processes:  p.join()
    rs = []
    while not integrator.queue.empty():
        rs.append(integrator.queue.get())
    return sum(rs)
