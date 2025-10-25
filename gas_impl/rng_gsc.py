import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from time import localtime, strftime
from functools import lru_cache
from scipy.stats import norm, chi2
from scipy.stats import skew, kurtosis, rv_continuous
import gc 
import inspect
from collections import OrderedDict
from typing import List, Optional


from .stable_count_dist import gsc_mu_by_f, gen_stable_count
from .fcm_dist import frac_chi_mean, frac_chi2_mean, fcm_mu_by_f, fcm_inverse, fcm_inverse_mu_by_f, fcm_moment
from .ff_dist import frac_f
from .utils import calc_stats_from_moments


# GSC is renamed to FG (frac_gamma) in 10/2025
# The code here still uses GSC for historical reasons

class RV_Simulator:
    def __init__(self, mu_fn, num_years, vol=0.85, s_prec=2, 
                 is_ratio=True, W1_rv=None):
        self.mu_fn = mu_fn
        self.num_years = num_years
        self.vol = vol
        self.s_prec = s_prec
        self.is_ratio = is_ratio
        self.W1_rv = W1_rv
        self.dt = 1.0/365.0  # one day
        self.size = int(np.floor(self.num_years/self.dt))
        self.rs = pd.DataFrame()  # result set
        
        self.gsc: Optional[rv_continuous] = None  # it will be created in subclass

    def initialize_mu_cache(self, max_s=10.0):
        for s in np.linspace(0.001, max_s, num = int(max_s * 1000)):
            self.mu_fn(round(s,self.s_prec))

    def gsc_first_moment(self):
        assert self.gsc is not None
        return self.gsc.moment(1.0)  # type: ignore

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
                continue
            s = round(s * 0.95, 4)

        raise Exception(f"ERROR: fail to locate lowest s")

    def _get_W1(self):
        if self.W1_rv is None:
            W1 = norm.rvs(size=self.size)
        else:
            W1 = self.W1_rv.rvs(size=self.size)
        return np.array(W1)  # type: ignore 

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
        W1 = self._get_W1()

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

    @property
    def x_squared(self):
        return np.square(self._get_rs('x'))

    @property
    def s_squared(self):
        return np.square(self._get_rs('s'))

    # -------------------------------------
    def _stats_of(self, z):
        return {
            'mean': np.mean(z),
            'var':  np.var(z),  # type: ignore
            'skew': skew(z), 
            'kurtosis': kurtosis(z),
        }

    def stats_of_s(self):
        return self._stats_of(self.s)

    def stats_of_x(self):
        return self._stats_of(self.x)

    def stats_of_x_squared(self):
        return self._stats_of(self.x_squared)

    def combine_stats(self, stats_of_x, dist_stats) -> pd.DataFrame:
        df = pd.DataFrame(data=[
            pd.Series(data=stats_of_x, name='simulated'),
            pd.Series(data=dist_stats, name='analytic'),
        ])
        df.loc['error',:] = (df.loc['simulated'] - df.loc['analytic']).abs() / df.loc['analytic']

        col = 'kurtosis'  # kurtosis needs a different rule due to 3.0 offset
        df.loc['error', col] = abs(df.loc['simulated', col] - df.loc['analytic', col]) / (df.loc['analytic', col] + 3.0) # type: ignore
        return df


# ---------------------------------------------------------------------
def gsc_stats(g):
    # g: gen_stable_count
    m = [ g.moment(n*1.0) for n in [0,1,2,3,4]]
    return calc_stats_from_moments(m)



class GSC_RV_Simulator(RV_Simulator):
    def __init__(self, alpha_gsc, sigma, d, p, 
                 num_years=200000, is_ratio=True, initialize=True,
                 W1_rv=None):
        self.alpha_gsc = alpha_gsc
        self.sigma = sigma
        self.d = d
        self.p = p
        self.s_prec = 2  # this affects speed a lot
        self.gsc = self.create_gsc()  # type: ignore

        self.std_mu.cache_clear() 
        mu_fn = lambda x: self.std_mu(x)
        super().__init__(mu_fn, num_years, s_prec=self.s_prec, is_ratio=is_ratio, W1_rv=W1_rv)
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
        self.gsc_stats_df = df = self.combine_stats(self.stats_of_s(), self.gsc_stats())
        return df

    def validate_gsc_stats(self, skip_skew: bool = False, dist_name="GSC"):
        # dist_name to allow FCM to override
        gc.collect()
        df = self.get_gsc_stats()
        assert isinstance(df, pd.DataFrame)
        print(df)
        if self.d < 0:
            assert df.loc['error', 'mean'] < 0.08  # type: ignore
            print(f"OK: {dist_name} assertion on mean passed, skip rest due to negative d ************** ")
            return

        assert df.loc['error', 'mean'] < 0.05  # type: ignore
        assert df.loc['error', 'var']  < 0.05  # type: ignore
        if not skip_skew:
            assert df.loc['error', 'skew'] < 0.10  # type: ignore
        assert df.loc['error', 'kurtosis'] < 0.20  # type: ignore
        print(f"OK: {dist_name} assertion passed ************** ")

    def _calc_pdf(self, u, pdf) -> pd.DataFrame:
        df = pd.DataFrame({'u': u})
        df['p'] = df.u.parallel_apply(pdf)  # type: ignore
        return df

    def _plot_setup(self, ax, title, xlabel, x_min, x_max, ylabel="density", legend_loc="upper right"):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim([x_min, x_max])
        ax.legend(loc=legend_loc)
        ax.set_title(title)

    def plot_gsc(self, ax, s_max=None, title="GSC histogram (red)"):
        mn = np.mean(self.s)
        sd = np.var(self.s)**0.5  # type: ignore
        
        if s_max is None: s_max = mn + sd*5
        
        df = self._calc_pdf(np.linspace(0.01, s_max, num=501), self.gsc.pdf)  # type: ignore 
        ax.plot(df.u, df.p, c="blue", lw=1.5, linestyle='--', label=f"theoretical")

        y = ax.hist(self.s, bins=200, color="red", range=(0, s_max), density=True)
        self._plot_setup(ax, title, "s", x_min=0, x_max=s_max)

    def plot_mu(self, ax, mu_max=None, s_max=None, title="mu used in GSC simulation"):
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
        ax.set_title(title)


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
class FCM_RV_Simulator(GSC_RV_Simulator):
    def __init__(self, alpha, k, 
                 num_years=100000, is_ratio=True, initialize=True,
                 is_fcm2=False,  # tweak it to simulate FCM2
                 W1_rv=None):
        self.alpha = alpha
        self.k = k
        self.is_fcm2 = is_fcm2
        self.is_ratio = is_ratio
        self.dist_name = "FCM" if not is_fcm2 else "FCM2"
        dist = self.create_gsc()
        pm = inspect.signature(gen_stable_count._pdf).bind('x', *dist.args, **dist.kwds).arguments
        print(pm)
        if not self.is_ratio: print("this is a product distribution")
        super().__init__(alpha_gsc=pm['alpha'], sigma=pm['sigma'], d=pm['d'], p=pm['p'], 
                         num_years=num_years, is_ratio=is_ratio, initialize=initialize,
                         W1_rv=W1_rv)
        if self.is_ratio:
            print(f"fcm first moment = {fcm_moment(1.0, self.alpha, self.k)}")
        self.s_squared_rv = frac_chi2_mean(self.alpha, self.k)  # this is only for FCM
        gc.collect()

    def create_gsc(self):
        if self.is_fcm2:
            assert self.is_ratio, "FCM2 only implemented for ratio"
            return frac_chi2_mean(alpha=self.alpha, k=self.k)

        if self.is_ratio:
            return frac_chi_mean(alpha=self.alpha, k=self.k)
        else:
            return fcm_inverse(alpha=self.alpha, k=-self.k)

    @lru_cache(maxsize=1000000)
    def std_mu(self, x):
        dz_ratio = 0.0001 if x/self.sigma > 0.2 else None


        if self.is_ratio:
            # this is for fcm, but not for inverse
            # if self.alpha == 1 and self.k == -1: 
            #     return 1.0/(2 * x**2) - 1.0 
            if not self.is_fcm2:
                return fcm_mu_by_f(x, dz_ratio=dz_ratio, alpha=self.alpha, k=self.k)  # this doesn't work when k < 0, sorry
            else:
                return fcm_mu_by_f(np.sqrt(x), dz_ratio=dz_ratio, alpha=self.alpha, k=self.k) * 0.5
        else:
            return fcm_inverse_mu_by_f(x, dz_ratio=dz_ratio, alpha=self.alpha, k=-self.k)
            # return super().std_mu(x)

    def validate_fcm_stats(self):
        # when k is large, skewness is small for FCM
        super().validate_gsc_stats(skip_skew=True, dist_name=self.dist_name)

    def plot_fcm(self, ax, s_max=None):
        super().plot_gsc(ax, s_max=s_max, title=f"{self.dist_name} histogram (red)")

    def plot_mu(self, ax, mu_max=None, s_max=None):
        super().plot_mu(ax, mu_max=mu_max, s_max=s_max, title=f"mu used in {self.dist_name} simulation")

    # for s = FCM, calculate s^2 = FCM2, as an indirect way to validate the PDF of FCM2
    def pdf_s_squared(self, y):
        assert not self.is_fcm2, "This is only for FCM, not FCM2"
        return self.s_squared_rv.pdf(y)  # type: ignore

    def plot_s_squared(self, ax, s2_max=None):
        assert not self.is_fcm2, "This is only for FCM, not FCM2"
        x2 = self.s_squared
        mn = np.mean(x2)
        sd = np.var(x2)**0.5  # type: ignore
        
        if s2_max is None: s2_max = mn + sd*5  # provide your value if this range is off, and often off
        df = self._calc_pdf(np.linspace(0.0, s2_max, num=201), self.pdf_s_squared)  # type: ignore 
        ax.plot(df.u, df.p, c="blue", lw=1.5, linestyle='--', label=f"theoretical")

        y = ax.hist(x2, bins=200, color="red", range=(0, s2_max), density=True)

        params = f"alpha {self.alpha} k {self.k}"
        title = f"FCM S^2 histogram (red, {params})"
        self._plot_setup(ax, title, "s_sqared", x_min=0, x_max=s2_max, ylabel="density (log)")
        ax.set_yscale('log')


# ---------------------------------------------------------------------
class FF_RV_Simulator:
    def __init__(self, sim: FCM_RV_Simulator, d: int):
        self.sim = sim
        self.d = d
        self.rv = frac_f(alpha=sim.alpha, k=sim.k, d=float(d))
        self.size = len(sim.s)
        self.u1 = chi2(df=d).rvs(size=self.size) / float(d)
        self.u2 = sim.s
        self.f = self.u1 / self.u2

    def validate_ff_stats(self):
        m1 = self.rv.moment(1.0)
        m2 = self.f.mean()
        err = abs(m1 - m2) / m1
        print(f"FF first moment: analytic {m1:.3f} simulated {m2:.3f} err {err:.3%}")
        assert err < 0.1
       
    def plot_ff_hist(self, ax):
        max_x = int(self.f.max() * 0.5)  # the right tail is too sparse for histogram
        f2 = self.f[ self.f < max_x ]
        ax.hist(f2, bins=int(max_x)*10, color="red", density=True)


    def plot_ff_pdf(self, ax, title, min_x, max_x, assert_max_pdf=1.0):
        x = np.linspace(min_x, max_x, 200)
        pdf = self.rv.pdf(x)  # type: ignore
        assert max(pdf) < assert_max_pdf

        ax.plot(x, pdf, color="blue")
        ax.set_xlabel('x')
        ax.set_ylabel('density')
        ax.set_xlim(0, max_x)
        ax.set_title(title)
