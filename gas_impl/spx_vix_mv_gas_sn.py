from functools import lru_cache
from typing import List, Optional, Dict 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import cov, histogram, histogram2d  # type: ignore
from numpy.linalg import det, inv  # type: ignore
from scipy.special import gamma
from scipy.integrate import quad
from scipy.optimize import root_scalar
from scipy.stats import skew, kurtosis
from diskcache import FanoutCache

from pyro.contrib.examples.finance import load_snp500


from adp_tf.jubilee.vix import VIX 
from adp_tf.jubilee.spx import SPX
from adp_tf.stable_count.fcm_dist import frac_chi_mean 

from .gas_dist import gas_sn
from .multivariate_sn import Multivariate_GAS_SN, Multivariate_GAS_SN_Adp_2D, _calc_rho


# -----------------------------------------------------------
cache = FanoutCache(size_limit=30*1000000)
#
@cache.memoize(typed=True, expire=7*86400, tag='mgas_spx_vix')
def _mgas_get_pdf1(g, x):
    assert len(x) == g.n
    assert isinstance(x[0], float)
    return g.pdf(list(x))


# -----------------------------------------------------------
class SPX_VIX_Data:
    def __init__(self, num_bins=200, end_date=pd.to_datetime('2024-03-31')):
        self.end_date = end_date
        self.vix_cutoff = 0.3
        self.spx_cutoff = 0.05
        self.num_bins = num_bins

        self.vix = VIX()
        self.vix.df['rtn'] = self.vix.df['px'].pct_change()
        self.spx = SPX()
        self.spx.df['rtn'] = self.spx.df['px'].pct_change()

        self.data_df = self.get_data_df()
        
        # cached statistics
        self.vix_kurtosis = kurtosis(self.vix.df.rtn.dropna())  # full sample 
        self.spx_kurtosis = kurtosis(self.spx.df.rtn.dropna())
        # full sample
        df = self.get_full_data_df()
        self.cov = cov(df.rtn_vix, y=df.rtn_spx)  # type: ignore
        self.rho = _calc_rho(self.cov)
        self._validate_total_density()
        
        self.contour_levels = [5, 10, 50, 200, 500, 800]

    def get_full_data_df(self):
        df = pd.merge(self.vix.df[['rtn']], self.spx.df[['rtn']], left_index=True, right_index=True, suffixes=['_vix', '_spx'])
        df = df.dropna().query("datadate <= @self.end_date")  # make it reproducible
        return df

    def get_data_df(self, scale=1.0, outer=False):
        # for display purpose, truncated data
        op = '<=' if not outer else '>='   # cutoff tails, or only tail
        df = self.get_full_data_df().query(f"rtn_vix.abs() {op} (@self.vix_cutoff * {scale}) and rtn_spx.abs() {op} (@self.spx_cutoff * {scale})") 
        return df

    def n2col(self, n: int):
        assert n in [0,1]
        return 'rtn_vix' if n == 0 else 'rtn_spx' 

    def ds(self, n: int) -> pd.Series: 
        return self.data_df[self.n2col(n)]

    @property
    def mean(self): 
        return [self.ds(0).mean(), self.ds(1).mean()]

    @lru_cache(maxsize=10)  # this is used a lot
    def hist2d_density(self):
        return histogram2d(self.ds(0), self.ds(1), bins=(self.num_bins, self.num_bins), density=True)

    @lru_cache(maxsize=10)
    def hist2d_peak_density(self, bins=4**2):
        counts = self.hist2d_density()[0]
        cnt = np.flip(np.sort(counts.flatten()))  # type: ignore
        return (cnt[:bins]).mean()

    @lru_cache(maxsize=10)  # this is used a lot
    def hist1d_density(self, n: int):
        df = self.data_df
        return histogram(df[self.n2col(n)], bins=self.num_bins, density=True)

    def _validate_total_density(self):
        counts, xbins, ybins = self.hist2d_density()
        dx = xbins[2] - xbins[1]
        dy = ybins[2] - ybins[1]
        total_density = counts.sum() * dx * dy
        assert abs(total_density - 1.0) < 1e-6

    # -----------------------------------------------------------
    # -----------------------------------------------------------
    # -----------------------------------------------------------
    def scatter_plot(self, ax):
        df = self.data_df
        ax.scatter(df.rtn_vix, df.rtn_spx, s=0.1)
        ax.set_xlim([-0.3, 0.3])
        ax.set_ylim([-0.05, 0.05])
        ax.set_xlabel("VIX return")
        ax.set_ylabel("SPX return")
        ax.set_title(f"VIX/SPX Scatter Plot (data rho={self.rho:.2f})")

    def contour_plot(self, ax):
        counts, xbins, ybins = self.hist2d_density()
        ax.contour(counts.transpose(),
            extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]],
            linewidths=2, 
            cmap = plt.cm.rainbow,  # type: ignore
            extend = 'max',
            levels = self.contour_levels
        )
        ax.set_xlabel("VIX return")
        ax.set_ylabel("SPX return")
        mx = self.hist2d_peak_density()
        ax.set_title(f"VIX/SPX Contour Plot (max={int(mx)})")

    def hist1d_plot(self, ax, n: int, label='histogram'):
        df = self.data_df
        return ax.hist(df[self.n2col(n)], bins=self.num_bins, density=True, label=label)

    def set_xlim_ylim(self, ax):
        xc = self.vix_cutoff
        ax.set_xlim([-xc, xc])
        yc = self.spx_cutoff
        ax.set_ylim([-yc, yc])


# --------------------------------------------------------------------
# --------------------------------------------------------------------
# models
model_config = { # third iteration, kurtosis is spot on, use full sample
    'gas1': {
        'alpha': [0.64, 0.875],
        'k':     [5.5,  3.202],
        'beta':  [0.0, 0.0],
    },
    'gas2': {  # second interation
        'alpha': [0.84, 0.62],
        'k':     [4.75, 5.50],
        'beta':  [0.0, 0.0],
    },
    'gas3': {  # first itereation
        'alpha': [0.75, 0.65],
        'k':     [5.50, 5.00],
        'beta':  [0.0, 0.0],
    },
    'gep1': {  # second iteration
        'alpha': [0.75, 0.62],
        'k':     [-4.30, -3.40],
        'beta':  [0.0, 0.0],
    },
    'gep2': {  # first shot
        'alpha': [0.75, 0.62],
        'k':     [-4.30, -3.40],
        'beta':  [0.0, 0.0],
    },
}


# --------------------------------------------------------------------
class SPX_VIX_Marginal_Dist:
    def __init__(self, data_obj: SPX_VIX_Data, length=40, model_name='gas1'):
        self.data_obj: SPX_VIX_Data = data_obj
        self.cov = self.data_obj.cov
        self.rho = self.data_obj.rho
        self.var0 = self.cov[0,0]
        self.var1 = self.cov[1,1]
        self.n = 2

        # --------------------------------------------------------------------
        self.model_name: str = model_name
        self.model = model_config[self.model_name]
        assert isinstance(self.model, Dict)
        self.alpha = self.model['alpha']
        self.k     = self.model['k']
        self.beta  = self.model['beta']

        self.gas_unit = [ gas_sn(alpha=self.alpha[i], k=self.k[i], beta=self.beta[i]) for i in range(self.n) ] 
        self.gas_sd = [ gas_sn(alpha=self.alpha[i], k=self.k[i], beta=self.beta[i], scale=self.cov[i,i]**0.5 * self.gas_rescale(i)) for i in range(self.n) ] 
        self.gas_pdf = [ self.create_1d_pdf_df(i) for i in range(self.n) ] 
        self.gas_kurtosis = [ gas_sn(self.alpha[i], self.k[i], self.beta[i]).stats('k') for i in range(self.n) ] 

        # assertion
        for i in range(self.n):
            p1 = self.gas_sd[i].stats('v')
            p2 = self.cov[i,i]
            assert abs(p1/p2 - 1.0) < 1e-3

        # --------------------------------------------------------------------
        self.ell_alpha = sum(self.alpha)/2
        self.ell_k = sum(self.k)/2
        self.ell_beta = self.beta

        self.ell_gas = self.find_ell_adjusted_instance()
        self.ell_gas_marginal = [ self.ell_gas.marginal_1d_rv(i) for i in range(self.n) ]   # TODO this is undefined yet

        self.adp_gas = self.find_adp_adjusted_instance()
        self.adp_gas_marginal = [ self.adp_gas.marginal_1d_rv(i) for i in range(self.n) ]   # TODO this is undefined yet
        print(f"mv adj factor: ell = {self.ell_adj_factor}, adp = {self.adp_adj_factor}")

        # --------------------------------------------------------------------
        # contour plot configuration
        scale = 1.0  # 2.0/3 if self.model_name.startswith('gsas') else 1.0
        self.x_max = self.data_obj.vix_cutoff * scale
        self.y_max = self.data_obj.spx_cutoff * scale
        self.length = length
        self.contour_levels = [5, 10, 50, 200, 500, 800]

        self.grid_x, self.grid_y = np.mgrid[  # type: ignore
            -self.x_max : self.x_max + 1e-6 : self.x_max*2/self.length, 
            -self.y_max : self.y_max + 1e-6 : self.y_max*2/self.length
            ]
        self.grid_pos = np.dstack((self.grid_x, self.grid_y))  # 2d (x,y)  # type: ignore
        n0, n1, _ = self.grid_pos.shape
        self.mv_pdf_df = pd.DataFrame(data = [{'i0': i0, 'i1': i1, 'xs': self.grid_pos[i0,i1]} for i0 in range(n0) for i1 in range(n1)])

    @property
    def ell_cov(self):  return self.ell_gas.cov
    
    @property
    def ell_rho(self):  return self.ell_gas.rho
    
    @lru_cache(maxsize=10)
    # TODO this is not correct, need to fix
    def ell_peak_density(self): return float(self.ell_gas.pdf(self.ell_gas.x0))  # type: ignore

    @property
    def adp_cov(self):  return self.adp_gas.cov
    
    @property
    def adp_rho(self):  return self.adp_gas.rho
    
    @lru_cache(maxsize=10)
    def adp_peak_density(self): return float(self.adp_gas.pdf(self.adp_gas.x0))  # type: ignore


    def create_1d_pdf_df(self, n: int):
        g = self.gas_sd[n]
        _, bins = self.data_obj.hist1d_density(n)
        df = pd.DataFrame(data={'x': bins})
        df['p1'] = df['x'].parallel_apply(lambda x: g.pdf(x))  # type: ignore

        def _assert_total_density(df):
            dx = df.x.diff()[1]
            ttl = df.p1.sum() * dx
            print(f"total density for {n} = {ttl}")
            # assert abs(ttl - 1.0) < 0.01, f"ERROR: total model density is not 1: n={n}, total={ttl}"

        _assert_total_density(df)
        return df

    def gas_rescale(self, i):
        return self.gas_unit[i].stats('v')**(-0.5)

    def ell_rescale(self):
        # TODO how to handle beta in ellipical case?
        return gas_sn(self.ell_alpha, self.ell_k, beta=0.0).stats('v')**(-0.5)

    def populate_mv_pdf_df(self, g, col):
        def _pdf1(x):
            assert len(x) == 2
            y = tuple([round(x[0],7), round(x[1],7)])
            return _mgas_get_pdf1(g, y)

        if col in self.mv_pdf_df.columns:
            ix = self.mv_pdf_df.eval(f"{col} > 0")  # okay rows
        else:
            ix = self.mv_pdf_df.eval("i0 < 0")  # redo all rows
        rows = len(self.mv_pdf_df.loc[~ix])
        if rows > 0:
            print(f"populating {rows} rows of {col}")
            self.mv_pdf_df.loc[~ix, col] = self.mv_pdf_df.loc[~ix]['xs'].parallel_apply(_pdf1)
 
    def get_arr_from_mv_pdf_df(self, col):
        n0, n1, _ = self.grid_pos.shape
        df = self.mv_pdf_df
        print(df.head(10))
        def _get_data(i0, i1): return df.query("i0 == @i0 and i1 == @i1").iloc[0][col]
        return [[ _get_data(i0,i1) for i1 in range(n1) ] for i0 in range(n0)]

    def _mv_contourf_plot(self, ax, pdf):
        # pdf has the same shape as pos since it is supposed to be pdf_fn(pos)
        x = self.grid_x
        y = self.grid_y
        ax.contourf(x, y, pdf, 
            cmap = plt.cm.rainbow,  # type: ignore
            extend = 'max',
            levels = self.contour_levels)
        ax.set_xlabel("VIX return")
        ax.set_ylabel("SPX return")
        self.data_obj.set_xlim_ylim(ax)

    def elliptical_plot(self, ax):
        col = 'ell_pdf'
        self.populate_mv_pdf_df(self.ell_gas, col)
        self._mv_contourf_plot(ax, self.get_arr_from_mv_pdf_df(col))
        mx = self.ell_peak_density()
        ax.set_title(f"2D GAS-SN Elliptical Contour (rho={self.ell_rho:.2f}, max={int(mx)})")
        ax.text(-0.25, -0.03, f"alpha={self.ell_alpha:.2f} k={self.ell_k:.2f}")

    def adaptive_plot(self, ax):
        col = 'adp_pdf'
        self.populate_mv_pdf_df(self.adp_gas, col)
        self._mv_contourf_plot(ax, self.get_arr_from_mv_pdf_df(col))
        mx = _mgas_get_pdf1(self.adp_gas, tuple(self.adp_gas.x0))
        ax.set_title(f"2D GAS-SN Adaptive Contour (rho={self.adp_rho:.2f}, max={int(mx)})")

    # --------------------------------------------------
    def vix_plot(self, ax1, ax2):
        n = 0
        df1 = self.gas_pdf[n]
        left_shift = 0.007
        x = df1.x + self.data_obj.mean[n] - left_shift
        
        def _plot_dist(ax):
            ax.plot(x, df1.p1, color='red', linewidth=1, label='GAS-SN')
            # ax.plot(df1.x + df.rtn_vix.mean() - 0.02, df1.p2, color='orange', linewidth=1, label='GAS')  # skewness shifts the mean
            ax.set_xlabel("daily return")
            ax.legend(loc="upper left")

        self.data_obj.hist1d_plot(ax1, n)
        _plot_dist(ax1)
        ax1.set_ylabel("density (in log scale)")
        ax1.set_yscale('log')
        ax1.set_title("VIX distribution (right skew)")
        
        self.data_obj.hist1d_plot(ax2, n)
        _plot_dist(ax2)
        ax2.set_ylabel("density")
        ax2.set_title(f"VIX alpha={self.alpha[n]:.2f} k={self.k[n]:.2f} rescale={self.gas_rescale(n):.3f}")
        ax2.text(-0.3, 6.0, f"sample kurtosis={self.data_obj.vix_kurtosis:.1f}\nmodel kurtosis={self.gas_kurtosis[n]:.1f}")
        return df1

    def spx_plot(self, ax1, ax2):
        n = 1
        df1 = self.gas_pdf[n]
        x = df1.x + self.data_obj.mean[n]

        def _plot_dist(ax):
            ax.plot(x, df1.p1, color='red', linewidth=1, label='GAS-SN')
            ax.set_xlabel("daily return")
            ax.legend(loc="upper left")

        self.data_obj.hist1d_plot(ax1, n)    
        _plot_dist(ax1)
        ax1.set_ylabel("density (in log scale)")
        ax1.set_yscale('log')
        ax1.set_title("SPX distribution (left skew)")
    
        self.data_obj.hist1d_plot(ax2, n)
        _plot_dist(ax2)
        ax2.set_ylabel("density")
        ax2.set_title(f"SPX alpha={self.alpha[n]:.2f} k={self.k[n]:.2f} rescale={self.gas_rescale(n):.3f}")
        ax2.text(-0.05, 45.0, f"sample kurtosis={self.data_obj.spx_kurtosis:.1f}\nmodel kurtosis={self.gas_kurtosis[n]:.1f}")
        return df1
    
    # --------------------------------------------------
    def create_adjusted_cov(self, adj_factor, mv_type):
        assert mv_type in ['ell', 'adp']
        v0 = (self.cov[0,0] * (1.0 - adj_factor))**0.5 * (self.gas_rescale(0) if mv_type == 'adp' else self.ell_rescale())
        v1 = (self.cov[1,1] * (1.0 - adj_factor))**0.5 * (self.gas_rescale(1) if mv_type == 'adp' else self.ell_rescale())
        rho = self.rho * (1.0 + adj_factor)
        new_cov = np.array([
            [ v0**2, v0*v1*rho ],
            [ v0*v1*rho, v1**2 ],
        ])
        return new_cov
    
    def create_adjusted_mv_instance(self, adj_factor, create_mv_fn, mv_type):
        return create_mv_fn(self.create_adjusted_cov(adj_factor, mv_type))

    def eval_mv_adj_distance(self, adj_factor, create_mv_fn, mv_type):
        mgas = self.create_adjusted_mv_instance(adj_factor, create_mv_fn, mv_type)
        p1 = mgas.pdf_at_zero()
        p2 = self.data_obj.hist2d_peak_density() * (1.0 - adj_factor)
        return p1 - p2

    def create_ell_mv_fn(self, cov): return Multivariate_GAS_SN(cov=cov, alpha=self.ell_alpha, k=self.ell_k, beta=self.ell_beta)
    def create_adp_mv_fn(self, cov): return Multivariate_GAS_SN_Adp_2D(cov=cov, alpha=self.alpha, k=self.k, beta=self.beta)

    def find_ell_adjusted_instance(self):
        mv_fn = lambda cov: self.create_ell_mv_fn(cov)
        def _eval_ell_distance(adj_factor):
            return self.eval_mv_adj_distance(adj_factor, mv_fn, mv_type='ell')

        self.ell_adj_factor = round(root_scalar(_eval_ell_distance, x0=0.05, x1=0.15).root, 4)
        return self.create_adjusted_mv_instance(self.ell_adj_factor, mv_fn, mv_type='ell')

    def find_adp_adjusted_instance(self):
        mv_fn = lambda cov: self.create_adp_mv_fn(cov)
        def _eval_adp_distance(adj_factor):
            return self.eval_mv_adj_distance(adj_factor, mv_fn, mv_type='adp')

        self.adp_adj_factor = round(root_scalar(_eval_adp_distance, x0=0.05, x1=0.15).root, 4)
        return self.create_adjusted_mv_instance(self.adp_adj_factor, mv_fn, mv_type='adp')
