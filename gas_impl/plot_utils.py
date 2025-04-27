# this file contains small utility functions for plotting
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import inspect
from typing import Optional, List

from numpy import histogram, histogram2d  # type: ignore
from scipy.stats import skew, kurtosis

from .mle_gas_sn import LLK_Calculator
from .mle_gas_sn_2d import LLK_Calculator_2D

from .gas_sn_dist import gas_sn, GAS_SN
from .ff_dist import FracF_PPF  # type: ignore


class TwoSided_Plots:
    def __init__(self, lkc: LLK_Calculator, dist_name: str, data_name: str, num_bins=200):
        self.lkc: LLK_Calculator = lkc
        self.data: pd.Series = lkc.data 
        self.sorted_data: pd.Series = np.sort(self.data)  # type: ignore
        self.size = len(self.data)
        self.rv = lkc.rv
        self.dist_name = dist_name
        self.data_name = data_name
        self.num_bins = num_bins

        cnt, bins = histogram(self.data, bins=self.num_bins, density=True)
        self.data_pdf = cnt


        self.df = self.calc_pdf()  # this takes time
        self.qp: Optional[PPF_Quantile_Plots]
        
    def rv_stats(self):
        if isinstance(self.rv, GAS_SN):
            return self.rv.stats_mvsk()  # this only works for my GAS_SN class
        else:
            # this only works for rv_continuous
            return self.rv.stats(moments='mvsk')  # type: ignore
    
    def rv_params(self):
        rv = self.rv
        if isinstance(self.rv, GAS_SN):
            return {'alpha': rv.alpha, 'k': rv.k, 'beta': rv.beta}  # type: ignore
        else:
            # this only works for rv_continuous
            return inspect.signature(gas_sn._pdf).bind('x', *rv.args, **rv.kwds).arguments  # type: ignore

    @property
    def alpha(self):
        return self.rv_params()['alpha']

    @property
    def k(self):
        return self.rv_params()['k']
    
    @property
    def beta(self):
        return self.rv_params()['beta']

    def plot_hist(self, ax):
        ax.axvline(x=self.data.mean(), color='blue', linestyle='--', linewidth=1)
        return ax.hist(self.data, bins=self.num_bins, density=True, label='histogram')

    def calc_pdf(self) -> pd.DataFrame:
        y = self.sorted_data
        def _data_cdf(x): return 1.0 * len(y[y <= x]) / self.size

        _, bins = histogram(self.data, bins=self.num_bins, density=True)
        df1 = pd.DataFrame(data={'x': bins})
        df1['observed_cdf'] = df1['x'].parallel_apply(lambda x: _data_cdf(x))
        df1['pdf'] = self.rv.pdf(df1['x'])  # type: ignore
        df1['cdf'] = self.rv.cdf(df1['x'])  # type: ignore
        df1['dx'] = df1.x.diff()
        df1.loc[0, 'dx'] = df1.loc[1, 'dx']  # type: ignore
        print('total pdf =', df1.eval("pdf * dx").sum())
        return df1
    

    def plot_dist(self, ax, y_min=5e-4):
        ax.plot(self.df.x, self.df.pdf, color='red', linewidth=1, label=self.dist_name)
        ax.axvline(x=self.rv.mean(), color='red', linestyle='--', linewidth=1)  # type: ignore
        ax.set_xlabel("Daily return")
        ax.legend(loc="upper right")
        # ax.set_xlim([-0.10, 0.10])
        max_p = max([ self.df.pdf.max(), self.data_pdf.max() ])
        ax.set_ylim([y_min, max_p * 1.1])

    def plot_two_pdf_charts(self, ax1, ax2, y_min=5e-4):
        self.plot_hist(ax1)
        self.plot_dist(ax1, y_min=y_min)
        ax1.set_ylabel("Density")
        ax1.set_title(f"{self.data_name}, data skew: {skew(self.data):.2f}, kurt: {kurtosis(self.data):.1f}, sd: {self.data.std():.2f}")
        
        st = self.rv_stats()
        self.plot_hist(ax2)
        self.plot_dist(ax2)
        ax2.set_yscale('log')
        ax2.set_ylabel("Density (in log scale)")
        ax2.set_title(f"{self.data_name} alpha={self.alpha:.2f} k={self.k:.1f} beta={self.beta:.2f} (sd: {st[1]**0.5:.2f} skew: {st[2]:.2f}, kurt: {st[3]:.1f})")

    def pp_plot(self, ax):
        pp_plot(ax, self.df.cdf, self.df.observed_cdf)
        ax.set_title(f"{self.data_name} PP-plot (mean: {self.data.mean():.2f} vs th {self.rv.mean():.2f})")  # type: ignore
        
    def squared_qq_plot(self, ax):
        self.qp = PPF_Quantile_Plots(self.lkc.get_squared_ppf_rv())
        self.qp.qq_plot(ax)
        ax.set_title(f"{self.data_name} Squared QQ-plot")


class PPF_Quantile_Plots:
    def __init__(self, ppf_rv):
        self.ppf_rv: FracF_PPF = ppf_rv  # this should contains the data and the rv
        print(f"analyzing quantiles in init")
        self.observed_quantiles, self.theoretical_quantiles = self.ppf_rv.analyze_quantiles()  # to warm it up
        assert isinstance(self.observed_quantiles, np.ndarray) 
        assert isinstance(self.theoretical_quantiles, np.ndarray)
        print(f"finished quantile analysis")

    def qq_plot(self, ax, max_x: Optional[float] = None, min_x: Optional[float] = None, log_scale=False):
        assert isinstance(self.theoretical_quantiles, np.ndarray)
        if max_x is None:
            max_x = max(self.theoretical_quantiles) * 1.1
            assert isinstance(max_x, float) and max_x >= 0, f"ERROR: max_x is {max_x} and should be >= 0"
        if min_x is None:
            min_x = min(self.theoretical_quantiles) * 0.9
            assert isinstance(min_x, float) and min_x >= 0, f"ERROR: min_x is {min_x} and should be >= 0"
            if log_scale and min_x < 1e-4: # avoid negative values
                min_x = 1e-4
        qq_plot(ax, self.theoretical_quantiles, self.observed_quantiles, max_x=max_x, min_x=min_x)
        if log_scale:
            ax.set_xscale('log')
            ax.set_yscale('log')

        rv_mean = self.ppf_rv.moment(1)
        data_mean = np.mean(self.observed_quantiles)
        ax.axvline(x=rv_mean, color='r', linestyle='--', linewidth=1.0, label=f"theoretical mean: {rv_mean:.2f}")
        ax.axhline(y=data_mean, color='r', linestyle='--', linewidth=1.0, label=f"observed mean: {data_mean:.2f}")
        ax.legend(loc='lower right')

    def pp_plot(self, ax):
        observed_cdf, theoretical_cdf = self.ppf_rv.analyze_cdf()  # to warm it up
        pp_plot(ax, theoretical_cdf, observed_cdf)


# ----------------------------------------------------------------------------
# validation based on Azzalini (2013)
# These are useful for plotting the quadratic form
# ----------------------------------------------------------------------------
def qq_plot(ax, theoretical_quantiles, observed_quantiles, max_x, min_x: Optional[float] = None):
    ax.scatter(theoretical_quantiles, observed_quantiles, alpha=0.75, s=3)
    ax.plot([0, max_x], [0, max_x], 'r--', lw=0.5)  # 45-degree reference line
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Observed Quantiles")
    min_x = 0.0 if min_x is None else min_x
    ax.set_xlim([min_x, max_x])
    ax.set_ylim([min_x, max_x])
    ax.grid(True)


def pp_plot(ax, theoretical_cdf, emperical_cdf):
    ax.plot(theoretical_cdf, emperical_cdf, c="blue", lw=1.5, alpha=0.75)
    ax.plot([0, 1], [0, 1], 'r--', lw=0.5)  # 45-degree line
    ax.set_xlabel("Theoretical CDF")
    ax.set_ylabel("Empirical CDF")
    ax.grid(True)


def _ax_lines(ax, x, y):
    ax.axvline(x=x, color='r', linestyle='--', linewidth=0.5)
    ax.axhline(y=y, color='r', linestyle='--', linewidth=0.5)


# ----------------------------------------------------------------------------
class TwoDim_Plots:
    def __init__(self, lkc: LLK_Calculator_2D, data_name=['VIX', 'SPX'], num_bins=200, squared_multiplier=1.0):
        self.lkc: LLK_Calculator_2D = lkc
        self.data: pd.DataFrame = lkc.data 
        self.size = len(self.data)
        self.rv = lkc.rv
        self.data_name: List[str] = data_name
        self.num_bins = num_bins
        self.squared_multiplier = squared_multiplier
        self.contour_len = 40
        
        self.data_pdf, self.xbins, self.ybins = histogram2d(self.data.x, self.data.y, bins=(self.num_bins, self.num_bins), density=True)
        print(f"squared_multiplier = {self.squared_multiplier}")
        self.ppf_plots = PPF_Quantile_Plots(self.lkc.get_squared_ppf_rv(multiplier=self.squared_multiplier))

    @property
    def theoretical_corr(self):
        return self.lkc.rv.var2corr()[0,1]  # type: ignore

    @property
    def empirical_corr(self):
        return self.lkc.empirical_corr  # type: ignore

    def _set_labels(self, ax):
        ax.set_xlabel(f"{self.data_name[0]} return")
        ax.set_ylabel(f"{self.data_name[1]} return")

    def _plot_slope(self, ax, corr, color, type, min_x=-4.0, max_x=4.0):
        x = np.linspace(min_x, max_x, 100)
        y = corr * x
        ax.plot(x, y, color=color, linestyle='--', linewidth=0.75, label=f"{type} corr: {corr:.3f}")
    
    def _plot_corr_lines(self, ax):
        self._plot_slope(ax, corr=self.empirical_corr, color='red', type='empirical')
        self._plot_slope(ax, corr=self.theoretical_corr, color='blue', type='theoretical')
        
    def scatter_plot(self, ax):
        df = self.data
        ax.scatter(df.x, df.y, s=0.1)
        self._set_labels(ax)
        self._plot_corr_lines(ax)
        _ax_lines(ax, 0, 0)
        ax.set_title(f"{self.data_name[0]}/{self.data_name[1]} Rtn Scatter Plot (corr: {self.empirical_corr:.3f} vs th {self.theoretical_corr:.3f})")
        ax.legend(loc='upper right')

    def contour_plot(self, ax):        
        ax.contour(self.data_pdf.transpose(),
            extent = [self.xbins[0], self.xbins[-1], self.ybins[0], self.ybins[-1]],
            linewidths=1.5, 
            cmap = plt.cm.rainbow,  # type: ignore
            levels = np.array([5, 20, 50, 100, 200, 500, 800]) / 1000.0
        )
        self._set_labels(ax)
        self._plot_corr_lines(ax)
        _ax_lines(ax, 0, 0)
        ax.set_title(f"{self.data_name[0]}/{self.data_name[1]} Rtn Contour Plot")
        ax.legend(loc='upper right')

    def quad_pp_plot(self, ax):
        m = self.ppf_plots
        m.pp_plot(ax)
        type='Ell' if not self.lkc.is_adp else 'Adp'
        ax.set_title(f"{type} Quad Form PP-Plot (th corr: {self.theoretical_corr:.3f})")

    def quad_qq_plot(self, ax, log_scale=False):
        m = self.ppf_plots
        m.qq_plot(ax, log_scale=log_scale)
        type = 'Ell' if not self.lkc.is_adp else 'Adp'
        msg = "(log scale)" if log_scale else ""
        ax.set_title(f"{type} Quad Form QQ-Plot {msg}")

    def theoretical_countour_plot(self, ax, type='Theoretical', x_max=3.0, y_max=3.0):
        x, y = np.mgrid[  # type: ignore
            -x_max : x_max : (x_max*2/self.contour_len), 
            -y_max : y_max : (y_max*2/self.contour_len)
        ]
        pos = np.dstack((x, y))  # type: ignore
        # ax.set_aspect('equal')
        ax.contourf(x, y, self.lkc.rv.pdf(pos))
        self._set_labels(ax)
        ax.set_title(f"{type} Contour (th corr: {self.theoretical_corr:.3f})")

    def plot_elliptical(self, ax, type='GAS-SN Elliptical', x_max=3.0, y_max=3.0):
        self.theoretical_countour_plot(ax, type=type, x_max=x_max, y_max=y_max)


    def plot_adaptive(self, ax, type='GAS-SN Adaptive', x_max=3.0, y_max=3.0):
        self.theoretical_countour_plot(ax, type=type, x_max=x_max, y_max=y_max)
