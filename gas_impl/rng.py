import re
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from time import localtime, strftime
from functools import lru_cache
from scipy.stats import skewnorm
import inspect
from collections import OrderedDict
from typing import List
from abc import abstractmethod

from scipy.stats import rv_continuous

from .gas_dist import gsas, levy_stable_from_feller, gsas_squared
from .gas_sn_dist import gas_sn
from .gexppow_dist import gexppow
from .utils import calc_stats_from_moments

from .rng_gsc import FCM_RV_Simulator



# ---------------------------------------------------------------------
class Two_Sided_RV_Simulator(FCM_RV_Simulator):
    # this include GEP (k > 0) by setting is_ratio = False
    def __init__(self, two_sided_name: str, 
                 alpha: float, k: float,
                 gamma: float = 0.0, 
                 beta: float = 0.0,
                 num_years=100000, is_ratio=True, initialize=True,
                 W1_rv=None):
        self.gamma = gamma  # TODO this is reserved for lihn_stable simulation
        self.beta = beta  # for GAS-SN

        if W1_rv is None and self.beta != 0:
            W1_rv = skewnorm(beta)  # beta is primarily used to inform skewnorm() usage
            
        super().__init__(alpha, k, 
                         num_years=num_years, is_ratio=is_ratio, initialize=initialize,
                         W1_rv=W1_rv)
        self.two_sided_rv: rv_continuous = self.create_two_sided_dist()
        self.two_sided_pm = self.get_two_sided_params()
        self.two_sided_name = two_sided_name 

    @abstractmethod
    def create_two_sided_dist(self) -> rv_continuous:  pass
    
    @abstractmethod
    def get_two_sided_params(self) -> OrderedDict:  pass

    def get_two_sided_params_by_pdf(self, pdf) -> OrderedDict:
        return inspect.signature(pdf).bind(
            'x', *self.two_sided_rv.args, **self.two_sided_rv.kwds).arguments  # type: ignore

    def two_sided_stats(self):
        m = [ self.two_sided_rv.moment(i) for i in range(5) ]
        return calc_stats_from_moments(m)

    def pdf_x_squared(self, y):
        # this is the default from general result
        g = self.two_sided_rv
        y2 = np.sqrt(y)
        return (g.pdf(y2) + g.pdf(-y2)) / (2 * y2)

    def get_combined_stats(self) -> pd.DataFrame:
        self.combined_stats_df = df = self.combine_stats(self.stats_of_x(), self.two_sided_stats())
        return df

    def validate_two_sided_stats(self, skip_skew=False):
        df = self.get_combined_stats()
        print(df)

        if not skip_skew:
            assert df.loc['error', 'skew'] < 0.10  # type: ignore

        if not self.is_ratio:  # gexppow, product dist
            assert self.k > 0
            assert df.loc['error', 'var']  < 0.05
            assert df.loc['error', 'kurtosis'] < 0.20
            print(f"OK: {self.two_sided_name} assertion passed ************** ")
            return

        if self.k < 0:
            assert df.loc['error', 'var']  < 0.10
        if self.k > 2:
            assert df.loc['error', 'var']  < 0.05

        if 0 < self.k <= 4:
            print(f"OK: {self.two_sided_name} assertion skipped, k is too small ************** ")
            return 

        assert df.loc['error', 'kurtosis'] < 0.20
        print(f"OK: {self.two_sided_name} assertion passed ************** ")

    def plot_two_sided(self, ax, x_max=None):
        mn = np.mean(self.x)
        sd = np.var(self.x)**0.5  # type: ignore
        
        if x_max is None: x_max = mn + sd*5  # provide your value if this range is off, and often off
        df = self._calc_pdf(np.linspace(-x_max, x_max, num=501), self.two_sided_rv.pdf)  # type: ignore 
        ax.plot(df.u, df.p, c="blue", lw=1.5, linestyle='--', label=f"theoretical")

        pm = self.two_sided_pm
        y = ax.hist(self.x, bins=200, color="red", range=(-x_max, x_max), density=True)

        params = f"alpha {self.alpha} k {pm['k']}"
        if 'gamma' in pm: params += f" gamma {pm['gamma']}"  # for lihn_stable
        if 'beta' in pm: params += f" beta {pm['beta']}"  # for gas_sn
        title = f"{self.two_sided_name} histogram (red, {params})"
        self._plot_setup(ax, title, "x", x_min=-x_max, x_max=x_max)

    def plot_x_squared(self, ax, x2_max=None):
        x2 = self.x_squared
        mn = np.mean(x2)
        sd = np.var(x2)**0.5  # type: ignore
        
        if x2_max is None: x2_max = mn + sd*5  # provide your value if this range is off, and often off
        df = self._calc_pdf(np.linspace(0.0, x2_max, num=201), self.pdf_x_squared)  # type: ignore 
        ax.plot(df.u, df.p, c="blue", lw=1.5, linestyle='--', label=f"theoretical")

        pm = self.two_sided_pm
        y = ax.hist(x2, bins=200, color="red", range=(0, x2_max), density=True)

        params = f"alpha {self.alpha} k {pm['k']}"
        title = f"{self.two_sided_name} X^2 histogram (red, {params})"
        self._plot_setup(ax, title, "x_sqared", x_min=0, x_max=x2_max, ylabel="density (log)")
        ax.set_yscale('log')


# ---------------------------------------------------------------------
class GAS_SN_RV_Simulator(Two_Sided_RV_Simulator):
    # this include GEP (k > 0) by setting is_ratio = False
    def __init__(self, alpha, k, beta=0.0, num_years=100000, is_ratio=True, initialize=True):
        super().__init__(self._get_dist_name(beta, is_ratio),
                         alpha, k, beta=beta,
                         num_years=num_years, is_ratio=is_ratio, initialize=initialize)

        self.x_squared_rv = gsas_squared(self.alpha, self.two_sided_pm['k'])
        
    def _get_dist_name(self, beta, is_ratio):
        # this is local
        dist_name = 'GAS-SN' if beta != 0 else 'GSaS'
        if not is_ratio:
            dist_name = 'GEP-SN' if self.beta != 0 else 'GEP'
        return dist_name
        
    def create_two_sided_dist(self):
        k = self.k * (1.0 if self.is_ratio else -1.0)
        return gas_sn(alpha=self.alpha, k=k, beta=self.beta)
    
    def pdf_x_squared(self, y):
        return self.x_squared_rv.pdf(y)  # type: ignore
    
    def get_two_sided_params(self) -> OrderedDict:
        return self.get_two_sided_params_by_pdf(gas_sn._pdf)

    def validate_gas_sn_stats(self, skip_skew=False):
        self.validate_two_sided_stats(skip_skew=skip_skew)

    def plot_gas_sn(self, ax, x_max=None):
        self.plot_two_sided(ax, x_max=x_max)


# ---------------------------------------------------------------------
class GSaS_RV_Simulator(Two_Sided_RV_Simulator):
    # this include GEP (k > 0) by setting is_ratio = False
    def __init__(self, alpha, k, num_years=100000, is_ratio=True, initialize=True):
        super().__init__(self._get_dist_name(is_ratio),
                         alpha, k, 
                         num_years=num_years, is_ratio=is_ratio, initialize=initialize)

        self.x_squared_rv = gsas_squared(self.alpha, self.two_sided_pm['k'])

    def _get_dist_name(self, is_ratio):
        # this is local
        return 'GSaS' if is_ratio else 'GEP'

    def create_two_sided_dist(self):
        k = self.k * (1.0 if self.is_ratio else -1.0)
        return gsas(alpha=self.alpha, k=k)

    def pdf_x_squared(self, y):
        return self.x_squared_rv.pdf(y)  # type: ignore

    def get_two_sided_params(self) -> OrderedDict:
        return self.get_two_sided_params_by_pdf(gsas._pdf)

    def two_sided_stats(self):
        s = super().two_sided_stats()
        s['mean'] = 0.0
        s['skew'] = 0.0
        return s

    def validate_gsas_stats(self):
       self.validate_two_sided_stats(skip_skew=True)

    def plot_gsas(self, ax, x_max=None):
        self.plot_two_sided(ax, x_max=x_max)

    def validate_2sided(self):
        # known cases
        x = 0.2
        if self.alpha == 1.0 and self.k > 0 and self.is_ratio:
            mu = self.k * (1-x**2)/2
            assert abs(self.std_mu(x) - mu) < 1e-4 
            print(f"student-t mu equiv validated")
            return

        if self.k == 1.0 and self.is_ratio:
            p1 = levy_stable_from_feller(self.alpha, 0.0).pdf(x)  # type: ignore
            p2 = self.gsas.pdf(x)  # type: ignore
            assert abs(p1 - p2) < 1e-4 
            print(f"sas pdf equiv validated")
            return

        if self.k == 1.0 and not self.is_ratio:
            p1 = gexppow(self.alpha, 1.0).pdf(x)  # type: ignore
            p2 = self.gsas.pdf(x)  # type: ignore
            assert abs(p1 - p2) < 1e-4 
            print(f"exppow (product) pdf equiv validated")
            return

        if self.k < 0:
            p1 = gexppow(self.alpha, abs(self.k)).pdf(x)  # type: ignore
            p2 = self.gsas.pdf(x)  # type: ignore
            assert abs(p1 - p2) < 1e-4 
            print(f"gexppow (ratio) pdf equiv validated")
            return

        return