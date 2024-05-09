import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from time import localtime, strftime
from functools import lru_cache
from scipy.stats import norm
from scipy.stats import skew, kurtosis
import gc 
import inspect
from collections import OrderedDict
from typing import List

from numba import jit

from .stable_count_dist import gsc_mu_by_f, gen_stable_count
from .fcm_dist import frac_chi_mean, fcm_mu_by_f, fcm_inverse, fcm_inverse_mu_by_f, fcm_moment
from .gas_dist import gsas, gsas_moment, levy_stable_from_feller
from .gexppow_dist import gexppow


class RV_Simulator:
    def __init__(self, mu_fn, num_years, vol=0.85, s_prec=2, is_ratio=True):
        self.mu_fn = mu_fn
        self.num_years = num_years
        self.vol = vol
        self.s_prec = s_prec
        self.is_ratio = is_ratio
        self.dt = 1.0/365.0  # one day
        self.size = int(np.floor(self.num_years/self.dt))
        self.rs = pd.DataFrame()  # result set

    def initialize_mu_cache(self, max_s=10.0):
        for s in np.linspace(0.001, max_s, num = int(max_s * 1000)):
            self.mu_fn(round(s,self.s_prec))

    def gsc_first_moment(self):
        return self.gsc.moment(1.0)

    def locate_lowest_s(self):
        lowest_s = 0.005
        s = self.gsc_first_moment()
        cnt = 0
        while s >= lowest_s/5 and cnt < 1e4:
            try:
                mu = self.mu_fn(s)
                cnt  = cnt + 1
                # print(f"{cnt}: s {s} mu {mu}")
                if abs(mu) > 200.0: return s
                if s <= lowest_s: return lowest_s  # default
            except:
                continue;
            s = round(s * 0.95, 4)

        raise Exception(f"ERROR: fail to locate lowest s")
            
    def generate(self):
        gc.collect()
        rng = np.random.default_rng()
        s = np.array([0.0] * self.size)
        x = np.array([0.0] * self.size)
        mu = np.array([0.0] * self.size)

        lowest_s = self.locate_lowest_s() 
        lowest_mu = self.mu_fn(lowest_s)
        print(f"gsc first moment = {self.gsc_first_moment()}")
        print(f"lowest_mu = {lowest_mu} at s = {lowest_s}")

        dW = rng.normal(size=self.size) * np.sqrt(self.dt)
        W1 = rng.normal(size=self.size)
        s[0] = self.gsc_first_moment() + rng.uniform(-1, 1) * 0.1  # initial value
        if s[0] <= lowest_s: s[0] = lowest_s
        x[0] = s[0] * W1[0]

        print(f"num_years = {self.num_years} size = {self.size} prec_s = {10**(-self.s_prec)}")
        sim_type = 'ratio' if self.is_ratio else 'product'
        print(f"RV to simulate a {sim_type} distribution: s0 = {s[0]}")

        # ---------------------------------------------------------------
        for i in np.arange(0, self.size-1):
            s_i = max([s[i], lowest_s])  # rounding for lru_cache to work
            mu_i = self.mu_fn(max([round(s_i,self.s_prec), lowest_s]))
            if not (abs(mu_i) >= 0): mu_i = lowest_mu
            drift = self.vol**2 * mu_i * self.dt
            rnd = self.vol * abs(s_i)**0.5 * dW[i] 
            ds = drift + rnd 
            mu[i] = mu_i
            # print(f"ds = {ds}")
            if np.isnan(ds):
                raise Exception(f"ERROR: NaN found, s = {s_i}, drift = {drift}, rnd = {rnd}")
            if s[i] + ds < lowest_s:
                ds = lowest_s + abs(rnd) # higher
                # print(f"WARN {i}: ds too volatile {ds}")

            s[i+1] = s[i] + ds
            if self.is_ratio:
                x[i+1] = W1[i+1] / s[i+1]
            else:  # product 
                x[i+1] = W1[i+1] * s[i+1]

            if i % (365 * 10000) == 1 and i > 1000:  # 10000 takes about 20 minutes
                tm = strftime("%Y-%m-%d-%H:%M:%S", localtime())
                print(f"{tm} {i:10d}: {100.0*i/self.size:.1f} pct, {i*self.dt:.1f} yr: " + 
                      f"s= {s[i]:.2f} (mean {np.mean(s[:i]):.4f}) ds= {ds:.6f} mu= {mu_i:.2f}")

        # done, store the results
        self.rs = pd.DataFrame(data = {'s': s, 'x': x, 'mu': mu})
        gc.collect()
        return self

    # -------------------------------------
    def _get_rs(self, col):
        assert not self.rs.empty
        ds = self.rs[col]
        assert isinstance(ds, pd.Series)
        return ds.to_numpy()

    @property
    def s(self):  return self._get_rs('s')

    @property
    def x(self):  return self._get_rs('x')

    @property
    def mu(self):  return self._get_rs('mu')

    # -------------------------------------
    def stats_of_s(self):
        return {
            'mean': np.mean(self.s),
            'var':  np.var(self.s), 
            'skew': skew(self.s), 
            'kurtosis': kurtosis(self.s),
        }

    def stats_of_x(self):
        return {
            'mean': np.mean(self.x),
            'var':  np.var(self.x), 
            'skew': skew(self.x), 
            'kurtosis': kurtosis(self.x),
        }


# ---------------------------------------------------------------------
def gsc_stats(g):
    # g: gen_stable_count
    m = [ g.moment(n*1.0) for n in [0,1,2,3,4]]
    gsc_var = m[2] - m[1]**2
    gsc_cm3 = m[3] -3*m[1]*m[2] + 2*m[1]**3
    gsc_cm4 = m[4] -4*m[1]*m[3] + 6*m[1]**2 *m[2] -3*m[1]**4

    return {
        'mean': m[1], 
        'var': gsc_var, 
        'skew': gsc_cm3 / gsc_var**1.5, 
        'kurtosis': (gsc_cm4 / gsc_var**2) - 3.0,
    }


class GSC_RV_Simulator(RV_Simulator):
    def __init__(self, alpha_gsc, sigma, d, p, num_years=200000, is_ratio=True, initialize=True):
        self.alpha_gsc = alpha_gsc
        self.sigma = sigma
        self.d = d
        self.p = p
        self.s_prec = 2  # this affects speed a lot
        self.gsc = self.create_gsc()

        self.std_mu.cache_clear() 
        mu_fn = lambda x: self.std_mu(x)
        super().__init__(mu_fn, num_years, s_prec=self.s_prec, is_ratio=is_ratio)
        if initialize:
            self.initialize_mu_cache(max_s=self.gsc.moment(1.0)*5.0)
            print(f"mu cache initialized")
    
    def create_gsc(self):
        return gen_stable_count(alpha=self.alpha_gsc, sigma=self.sigma, d=self.d, p=self.p)

    @lru_cache(maxsize=1000000)
    def std_mu(self, x):
        dz_ratio = 0.0001 if x > 0.5 else None
        return gsc_mu_by_f(x, dz_ratio=dz_ratio, alpha=self.alpha_gsc, sigma=self.sigma, d=self.d, p=self.p)

    def gsc_stats(self):
        return gsc_stats(self.gsc)

    def get_gsc_stats(self) -> pd.DataFrame:
        df = pd.DataFrame(data=[
            pd.Series(data=self.stats_of_s(), name='simulated'),
            pd.Series(data=self.gsc_stats(), name='analytic'),
        ])
        df.loc['error',:] = (df.loc['simulated'] - df.loc['analytic']).abs() / df.loc['analytic']
        df.loc['error','kurtosis'] = abs(df.loc['simulated','kurtosis'] - df.loc['analytic','kurtosis']) / (df.loc['analytic','kurtosis'] + 3.0)
        self.gsc_stats_df = df
        return df

    def validate_gsc_stats(self):
        gc.collect()
        df = self.get_gsc_stats()
        print(df)
        if self.d < 0:
            assert df.loc['error', 'mean'] < 0.08
            print("OK: GSC assertion on mean passed, skip rest due to negative d ************** ")
            return

        assert df.loc['error', 'mean'] < 0.05
        assert df.loc['error', 'var']  < 0.05
        assert df.loc['error', 'skew'] < 0.10
        assert df.loc['error', 'kurtosis'] < 0.20
        print("OK: GSC assertion passed ************** ")

    def plot_gsc(self, ax, s_max=None):
        mn = np.mean(self.s)
        sd = np.var(self.s)**0.5
        
        if s_max is None: s_max = mn + sd*5
        u = np.linspace(0.01, s_max, num=501)
        p = [self.gsc.pdf(x) for x in u]

        ax.plot(u, p, c="blue", lw=1.5, linestyle='--', label=f"theoretical")
        y = ax.hist(self.s, bins=200, color="red", range=(0, s_max), density=True)
        ax.set_xlabel("s")
        ax.set_ylabel("density")
        ax.set_xlim([0, s_max])
        ax.legend(loc="upper right")
        ax.set_title("GSC histogram (red)")

    def plot_mu(self, ax, mu_max=None, s_max=None):
        df = self.rs[['s', 'mu']].copy()
        df['s_rounded'] = df['s'].apply(lambda x: round(x,2))
        df2 = (
            df[['s_rounded', 'mu']].groupby(by=['s_rounded']).mean()
            .reset_index().rename(columns={'s_rounded': 's'})
        )
        if mu_max is not None: df2 = df2.query(f"mu <= {mu_max}")
        if s_max  is not None: df2 = df2.query(f"s <= {s_max}")
        ax.plot(df2.s, df2.mu, c="blue", lw=1.5, linestyle='--', label=f"mu")
        ax.legend(loc="upper right")
        ax.set_xlabel("s")
        ax.set_ylabel("mu(s)")
        ax.set_title("mu used in GSC simulation")


# ---------------------------------------------------------------------
class SC_RV_Simulator(GSC_RV_Simulator):
    def __init__(self, alpha, num_years=100000):
        self.alpha = alpha
        super().__init__(alpha_gsc=alpha, sigma=1.0, d=1.0, p=alpha, num_years=num_years)


class SV_RV_Simulator(GSC_RV_Simulator):
    def __init__(self, alpha, num_years=100000):
        self.alpha = alpha
        super().__init__(alpha_gsc=alpha/2, sigma=1.0/np.sqrt(2), d=1.0, p=alpha, num_years=num_years)


# ---------------------------------------------------------------------
class GSaS_RV_Simulator(GSC_RV_Simulator):
    # this include GEP (k > 0) by setting is_ratio = False
    def __init__(self, alpha, k, num_years=100000, is_ratio=True, initialize=True):
        self.alpha = alpha
        self.k = k
        self.is_ratio = is_ratio
        self.gsas = self.create_2sided_dist()
        dist = self.create_gsc()
        pm = inspect.signature(gen_stable_count._pdf).bind('x', *dist.args, **dist.kwds).arguments
        print(pm)
        if not self.is_ratio: print("this is a product distribution")
        super().__init__(alpha_gsc=pm['alpha'], sigma=pm['sigma'], d=pm['d'], p=pm['p'], 
                         num_years=num_years, is_ratio=is_ratio, initialize=initialize)
        if self.is_ratio:
            print(f"fcm first moment = {fcm_moment(1.0, self.alpha, self.k)}")
        gc.collect()

    def create_gsc(self):
        if self.is_ratio:
            return frac_chi_mean(alpha=self.alpha, k=self.k)
        else:
            return fcm_inverse(alpha=self.alpha, k=-self.k)
        
    def create_2sided_dist(self):
        if self.is_ratio:
            return gsas(alpha=self.alpha, k=self.k)
        else:
            return gsas(alpha=self.alpha, k=-self.k)
    
    def get_gsas_params(self) -> OrderedDict:
        return inspect.signature(gsas._pdf).bind('x', *self.gsas.args, **self.gsas.kwds).arguments

    @lru_cache(maxsize=1000000)
    def std_mu(self, x):
        dz_ratio = 0.0001 if x/self.sigma > 0.2 else None
        if self.is_ratio:
            # this is for fmc, but not for inverse
            # if self.alpha == 1 and self.k == -1: 
            #     return 1.0/(2 * x**2) - 1.0 

            return fcm_mu_by_f(x, dz_ratio=dz_ratio, alpha=self.alpha, k=self.k)  # this doesn't work when k < 0, sorry
        else:
            return fcm_inverse_mu_by_f(x, dz_ratio=dz_ratio, alpha=self.alpha, k=-self.k)
            # return super().std_mu(x)

    def gsas_stats(self):
        m2 = self.gsas.moment(2.0)
        m4 = self.gsas.moment(4.0)
        return {
            'mean': 0.0, 
            'var': m2, 
            'skew': 0.0, 
            'kurtosis': m4/m2**2 - 3.0,
        }

    def get_gsas_stats(self) -> pd.DataFrame:
        df = pd.DataFrame(data=[
            pd.Series(data=self.stats_of_x(), name='simulated'),
            pd.Series(data=self.gsas_stats(), name='analytic'),
        ])
        df.loc['error',:] = (df.loc['simulated'] - df.loc['analytic']).abs() / df.loc['analytic']
        df.loc['error','kurtosis'] = abs(df.loc['simulated','kurtosis'] - df.loc['analytic','kurtosis']) / (df.loc['analytic','kurtosis'] + 3.0)
        self.gsas_stats_df = df
        return df

    def validate_gsas_stats(self):
        df = self.get_gsas_stats()
        print(df)
        
        if not self.is_ratio:  # gexppow, product dist
            assert self.k > 0
            assert df.loc['error', 'var']  < 0.05
            assert df.loc['error', 'kurtosis'] < 0.20
            print("OK: GEP assertion passed ************** ")
            return

        if self.k < 0:
            assert df.loc['error', 'var']  < 0.10
        if self.k > 2:
            assert df.loc['error', 'var']  < 0.05

        if 0 < self.k <= 4:
            print("OK: GSaS assertion skipped, k is too small ************** ")
            return 

        assert df.loc['error', 'kurtosis'] < 0.20
        print("OK: GSaS assertion passed ************** ")

    def plot_gsas(self, ax, x_max=None):
        mn = np.mean(self.x)
        sd = np.var(self.x)**0.5
        
        if x_max is None: x_max = mn + sd*5  # provide your value if this range is off, and often off
        u = np.linspace(-x_max, x_max, num=501)
        p = [self.gsas.pdf(x) for x in u]

        pm = self.get_gsas_params()
        ax.plot(u, p, c="blue", lw=1.5, linestyle='--', label=f"theoretical")
        y = ax.hist(self.x, bins=200, color="red", range=(-x_max, x_max), density=True)
        ax.set_xlim([-x_max, x_max])
        ax.legend(loc="upper right")
        ax.set_xlabel("x")
        ax.set_ylabel("density")
        ax.legend(loc="upper right")
        params = f"alpha {self.alpha} k {pm['k']}"
        if self.is_ratio:
            ax.set_title(f"GSaS histogram (red, {params})")
        else:
            ax.set_title(f"GEP histogram (red, {params})")

    def validate_2sided(self):
        # known cases
        x = 0.2
        if self.alpha == 1.0 and self.k > 0 and self.is_ratio:
            mu = self.k * (1-x**2)/2
            assert abs(self.std_mu(x) - mu) < 1e-4 
            print(f"student-t mu equiv validated")
            return

        if self.k == 1.0 and self.is_ratio:
            p1 = levy_stable_from_feller(self.alpha, 0.0).pdf(x)
            p2 = self.gsas.pdf(x)
            assert abs(p1 - p2) < 1e-4 
            print(f"sas pdf equiv validated")
            return

        if self.k == 1.0 and not self.is_ratio:
            p1 = gexppow(self.alpha, 1.0).pdf(x)
            p2 = self.gsas.pdf(x)
            assert abs(p1 - p2) < 1e-4 
            print(f"exppow (product) pdf equiv validated")
            return

        if self.k < 0:
            p1 = gexppow(self.alpha, abs(self.k)).pdf(x)
            p2 = self.gsas.pdf(x)
            assert abs(p1 - p2) < 1e-4 
            print(f"gexppow (ratio) pdf equiv validated")
            return

        return