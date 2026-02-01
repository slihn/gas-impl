# One-dimensional MLE for the gas-sn model

from typing import Literal, Optional, Union
import pandas as pd
import numpy as np 
from datetime import datetime
import json

from numpy import histogram  # type: ignore
from scipy.stats import skew, kurtosis

from .fcm_dist import frac_chi_mean, fcm_inverse
from .gas_sn_dist import GAS_SN


class LLK_Calculator:
    def __init__(self, data: pd.Series, compress_ratio=10.0, min_bins=100, lambda_pt=1.0, 
                 use_inverse: Optional[str] =None, use_999: bool = False):
        self.raw_data = data.copy()
        assert isinstance(data, pd.Series)
        self.raw_scale = self.raw_data.std()
        self.data = self.raw_data / self.raw_scale  # we really want to deal with standardized data
        self.lambda_pt = float(lambda_pt)  # regularization weight for 99-percentile matching
        self.use_999 = use_999  # regulate to 99.99 percentile if True
        
        # standardized the data to 1-std for MLE
        self.x = np.sort(np.array(self.data.copy()))  # type: ignore
        self.x_loc = self.x.mean()
        self.x_scale = self.x.std()  # this should be 1.0
        self.x_skew = skew(self.x)
        self.x_kurtosis = kurtosis(self.x)
        self.x_percentile_20 = np.percentile(self.x, 20)
        self.x_percentile_33 = np.percentile(self.x, 33)
        self.x_percentile_50 = np.percentile(self.x, 50)
        self.x_percentile_66 = np.percentile(self.x, 66)
        self.x_percentile_99 = np.percentile(self.x, 99)
        self.x_percentile_999 = np.percentile(self.x, 99.9)

        self.compress_ratio = float(compress_ratio)
        self.min_bins = int(min_bins)
        self.use_inverse: Optional[str] = use_inverse
        if self.use_inverse is not None:
            assert self.use_inverse in ['inv', 'char']

        self.delta4 = 1e-4
        self.delta5 = 1e-5

        # unrealistic initial values as placeholders
        # TODO should consider using self.hyper_params, like the 2D case
        self.alpha: float = 0.0
        self.k: float     = 0.0
        self.scale: float = 0.0  # scale and loc is relative to self.x, not input data
        self.loc: float = 0.0
        self.rv = None
        
        self._minus_log_likehood: Optional[float] = None

    @property
    def param_config(self) -> pd.DataFrame:
        ls = [
            {'name': 'alpha', 'delta': self.delta4, 'method': 'mul'},
            {'name': 'k',     'delta': self.delta4, 'method': 'mul'},
            {'name': 'scale', 'delta': self.delta4, 'method': 'mul'},
            {'name': 'loc',   'delta': self.delta5, 'method': 'delta'},
        ]
        df = pd.DataFrame(ls)
        assert df.set_index('name').index.is_unique
        return df

    def clone(self):
        mle =  LLK_Calculator(self.raw_data, compress_ratio=self.compress_ratio, min_bins=self.min_bins)
        self.copy_rv_to(mle)
        return mle

    def set_compress_ratio(self, compress_ratio):
        # this should handle the side effect risen from a new compress_ratio
        self.compress_ratio = float(compress_ratio)
        self._minus_log_likehood = None
        return self

    @property
    def bins(self):
        bins = int(len(self.x) / self.compress_ratio)
        if bins < self.min_bins: bins = self.min_bins
        return bins
    
    def get_histogram(self, positive_cnt=True) -> pd.DataFrame:
        _cnt, _xs = histogram(self.x, bins=self.bins, density=False)
        _xs_mean = ( _xs[1:] + _xs[:-1] ) / 2.0
        df = pd.DataFrame({'cnt': _cnt, 'x': _xs_mean})
        if positive_cnt:
            df = df.query("cnt > 0").copy()
        return df 
    
    def init_rv_from_gas_sn_class(self, g: GAS_SN):
        # this is a hack, simply use GAS_SN to hold parameters
        return self.init_rv(g.alpha, g.k, scale=g.scale, loc=g.loc)

    def init_rv(self, alpha: float, k: float, scale: float, loc: float=0.0):
        self.alpha = float(alpha)
        self.k     = float(k)
        self.scale = float(scale)
        self.loc   = float(loc)
        self.rv = self.get_rv(alpha, k, scale=scale, loc=loc)
        self._minus_log_likehood = None
        return self
    
    def get_rv(self, alpha: float, k: float, scale: float=1.0, loc: float=0.0):
        if self.use_inverse is not None:
            if self.use_inverse == 'inv':
                return fcm_inverse(alpha, k, scale=scale, loc=loc)
            elif self.use_inverse == 'char':
                return frac_chi_mean(alpha, -k, scale=scale, loc=loc)
            else:
                raise Exception(f"ERROR: unknown use_inverse: {self.use_inverse}")
        else:
            return frac_chi_mean(alpha, k, scale=scale, loc=loc)

    @property
    def rv_var(self) -> float:  return float(self.rv.stats(moments='v'))  # type: ignore

    @property
    def rv_skew(self) -> float:  return float(self.rv.stats(moments='s'))  # type: ignore

    @property 
    def rv_kurtosis(self) -> float:  return float(self.rv.stats(moments='k'))  # type: ignore
    
    def add_rv(self, lkc: 'LLK_Calculator', d_alpha=0.0, d_k=0.0, d_scale=0.0, d_loc=0.0) -> 'LLK_Calculator':
        self.init_rv(lkc.alpha + d_alpha, lkc.k + d_k, lkc.scale + d_scale, lkc.loc + d_loc)
        return self

    def copy_rv_to(self, mle):
        mle.alpha = self.alpha
        mle.k = self.k
        mle.scale = self.scale
        mle.loc = self.loc
        mle.rv = self.rv
        mle._minus_log_likehood = None
        return mle

    def get_hyper_params(self) -> pd.Series:
        return pd.Series({
            'alpha':  self.alpha,
            'k':      self.k,
            'scale':  self.scale, 
            'loc':    self.loc, 
        })

    def dump_hyper_params(self) -> str:
        parsed_json = json.loads(self.get_hyper_params().to_json())  # Convert to dict
        return json.dumps(parsed_json, indent=4, separators=(", ", ":\t"))  # Extra spaces after colon

    # ------------------------------------------------------------
    # L2 distance
    def L2_std(self) -> float:
        return (self.rv_var**0.5 - self.x_scale)**2

    # ------------------------------------------------------------
    def calc_minus_log_likehood(self, use_hist=True, force=False) -> float:
        # this MLLK is sum of log-likelihood, divided by the number of samples
        if self._minus_log_likehood is not None and not force: 
            return self._minus_log_likehood

        assert self.rv is not None
        def _log_pdf(x): return np.log(self.rv.pdf(x))  # type: ignore
        if not use_hist:
            self._minus_log_likehood = np.sum(_log_pdf(self.x)) * -1.0 / len(self.x)
            assert self._minus_log_likehood is not None
            return self._minus_log_likehood

        df = self.get_histogram()
        df['logpdf'] = df.x.parallel_apply(_log_pdf)
        self._minus_log_likehood = df.eval("cnt * logpdf").sum() * -1.0 / df.cnt.sum()
        assert self._minus_log_likehood is not None
        return self._minus_log_likehood
    
    def calc_mllk(self, use_hist=True, force=False):
        # just a simpler name
        return self.calc_minus_log_likehood(use_hist=use_hist, force=force)
    

    def calc_mllk_with_regularization(self, debug=False, iter=None):
        A = self.calc_mllk()
        B1 = self.lambda_pt * (self.rv.ppf(0.99)/self.x_percentile_99 - 1)**2  # type: ignore
        B2 = self.lambda_pt * (self.rv.ppf(0.66)/self.x_percentile_66 - 1)**2 * 10 # type: ignore
        B3 = self.lambda_pt * (self.rv.ppf(0.20)/self.x_percentile_20 - 1)**2 * 100  # type: ignore
        B4 = self.lambda_pt * (self.rv.ppf(0.999)/self.x_percentile_999 - 1)**2 * 100 if self.use_999 else 0.0  # type: ignore
        prefix = f"iter: {iter}, mllk-1" if iter is not None else "mllk-1"
        if debug: print(f"{prefix}: {A:.8f}, reg: ppf: 99 {B1:.8f} | {B2:.8f} {B3:.8f} | 99.9 {B4:.8f}")
        return A + B1 + B2 + B3 + B4

    def calc_gradient(self):
        delta4 = 1e-4
        delta5 = 1e-5

        r = pd.Series({
            'alpha': self.alpha, 'k': self.k, 'scale': self.scale, 'loc': self.loc,
            'd_alpha': self.alpha * delta4,
            'd_k':     self.k * delta4,
            'd_scale': delta5,
            'd_loc':   delta5,
            })

        def _calc_mllk(d_alpha=0.0, d_k=0.0, d_scale=0.0, d_loc=0.0):
            try:
                self.init_rv(r['alpha'] + d_alpha, r['k'] + d_k, r['scale'] + d_scale, r['loc'] + d_loc)  # type: ignore
                mllk = self.calc_mllk_with_regularization()
            except Exception as e:
                print(f"Exception in _calc_mllk: {e}")
                mllk = np.inf
            return mllk

        r['mllk'] = _calc_mllk()
        r['mllk_reg'] = 0.0
        r['grad_alpha'] = (_calc_mllk(d_alpha = r.d_alpha) - r.mllk) / r.d_alpha 
        r['grad_k']     = (_calc_mllk(d_k =     r.d_k)     - r.mllk) / r.d_k
        r['grad_scale'] = (_calc_mllk(d_scale = r.d_scale) - r.mllk) / r.d_scale
        r['grad_loc']   = (_calc_mllk(d_loc =   r.d_loc)   - r.mllk) / r.d_loc

        # ensure calc_gradient() does not change the RV state of the object at the end
        self.init_rv(r.alpha, r.k, r.scale, r['loc'])  # type: ignore
        return r


class MLE:
    def __init__(self, lkc: LLK_Calculator, 
                 kurt_range=[-1000.0, 1000.0], skew_range=None, 
                 max_iter: int = 60 * 4,  # 200 minutes
                 ):
        self.lkc = lkc
        self.lkc2 = lkc.clone()  # as a storage place during the gradient descent phase

        # the more we know, the better we confine the ranges
        self.kurt_range = kurt_range
        if skew_range is not None:
            self.skew_range = skew_range
        else:
            max_skew = 50.0
            skew = self.lkc.data.skew()
            assert isinstance(skew, float)
            if skew >= 0.0:
                self.skew_range = [0.0, max_skew]
            else:
                self.skew_range = [-max_skew, 0.0]
        
        # for gradient descent
        self.learning_rate: float = 0.01  # initial learning rate
        self.min_learning_rate: float = 1e-4 # stop when it's too small
        self.max_learning_rate: float = 1.0  # cap it so it won't jump randomly
        self.learning_bump: float = 0.4
        self.max_iter: int = max_iter  # 40 minutes
        self.max_alpha_move: float = 0.02  # don't allow alpha to jump too big

        # results are stored below
        self.scan_result = pd.DataFrame()  # stores the result of scan_param_space()
        self.descent_history = pd.DataFrame()  # stores the result of gradient descent

    # --------------------------------------------------------------------------------
    def get_param_space(self, min_alpha, max_alpha, num_alpha: int, min_k, max_k, num_k: int):
        df = pd.DataFrame([
            { 'alpha': alpha, 'k': k}
            for alpha in np.linspace(min_alpha, max_alpha, num=num_alpha) 
            for k in np.linspace(min_k, max_k, num=num_k)
        ])
        return df
    
    def get_param_space_for_vix(self):
        return self.get_param_space(0.3, 1.75, 30, 1.5, 7.0, 30)

    def get_allowed_param_space(self, param_space: pd.DataFrame):
        min_kurt, max_kurt = self.kurt_range
        min_skew, max_skew = self.skew_range
        df = param_space.copy()
        df['scale'] = self.lkc.x_scale  # this is just 1.0
        df['mean'] = df.apply(lambda r: self.lkc.get_rv(r.alpha, r.k).stats(moments='m'), axis=1)
        df['kurtosis'] = df.apply(lambda r: self.lkc.get_rv(r.alpha, r.k).stats(moments='k'), axis=1)
        df['skew'] = df.apply(lambda r: self.lkc.get_rv(r.alpha, r.k).stats(moments='s'), axis=1)
        df['loc'] = self.lkc.x_loc - df['mean']
        print(f"initial {len(df)} combinations")
        print(f"filtering kurtosis in [{min_kurt}, {max_kurt}], skew in [{min_skew}, {max_skew}]")

        df = (
            df.query("kurtosis >= @min_kurt and kurtosis <= @max_kurt")
            .query("skew >= @min_skew and skew <= @max_skew")
        )
        print(f"executable {len(df)} combinations")
        return df.copy()
    
    def scan_param_space(self, param_space: pd.DataFrame, max_combo=1000):
        df = self.get_allowed_param_space(param_space)
        if len(df) > max_combo:  raise ValueError(f"too many combinations: {len(df)} > {max_combo}")

        def _calc_mllk(r: pd.Series) -> float:
            self.lkc.init_rv(r.alpha, r.k, scale=r['scale'], loc=r['loc'])  # type: ignore
            return self.lkc.calc_minus_log_likehood()

        df['mllk'] = df.apply(_calc_mllk, axis=1)
        self.scan_result = df
        return df
    
    def get_best_scan_result(self) -> pd.Series:
        assert len(self.scan_result) > 0
        return self.scan_result.sort_values('mllk').head(1).iloc[0]

    # --------------------------------------------------------------------------------
    def make_next_move(self, gr: pd.Series, learning_rate: float) -> float:
        next_move = gr[[x for x in gr.index.values if x.startswith('grad')]]\
            .apply(lambda x: x * (np.random.rand() * 0.5 + 0.5) * learning_rate * -1.0)
        self.lkc.copy_rv_to(self.lkc2)
        try:
            self.lkc.init_rv(
                self.lkc.alpha + next_move.grad_alpha,  # type: ignore
                self.lkc.k     + next_move.grad_k,      # type: ignore
                self.lkc.scale + next_move.grad_scale,  # type: ignore
                self.lkc.loc   + next_move.grad_loc)    # type: ignore
            mllk = self.lkc.calc_mllk_with_regularization()
        except Exception as e:
            print(f"Exception in make_next_move: {e}")
            mllk = np.inf
        return mllk
    
    def append_descent_history(self, gr: pd.Series) -> pd.DataFrame:
        gr['timestamp'] = datetime.now()
        df = pd.DataFrame(data= [gr])
        if len(self.descent_history) == 0:
            self.descent_history = df
        else:
            self.descent_history = pd.concat([self.descent_history, df])
        return self.descent_history
    
    def gradient_descent(self, accept_max_iter=True) -> pd.Series:
        self.lkc.copy_rv_to(self.lkc2)
        print(f"start time: {datetime.now()}")
        print(f"lkc compress ratio: {self.lkc.compress_ratio}, bins: {self.lkc.bins}, max_iter: {self.max_iter}")

        def _calc_gradient(iter: int) -> pd.Series:
            gr = self.lkc.calc_gradient()
            gr['iter'] = iter
            print(f"iter: {iter}, mllkc: {gr.mllk:.8f} / {gr.mllk_reg:.8f} from alpha: {gr.alpha:.6f}, k: {gr.k:.6f}, scale: {gr.scale:.6f}, loc: {gr['loc']:.6f}")
            return gr
        
        def _gr2grad(gr: pd.Series, round=None) -> pd.Series:
            grad = gr[[x for x in gr.index.values if x.startswith('grad')]].copy()
            if round is not None:
                grad = grad.apply(lambda x: np.round(x, decimals=round))  # type: ignore
            return grad

        iter = 1 
        gr = _calc_gradient(iter)
        learning_rate = self.learning_rate
        while iter <= self.max_iter:
            mllk = gr.mllk

            # don't allow alpha to jump too big
            alpha_move = gr.grad_alpha * learning_rate
            if abs(alpha_move) > self.max_alpha_move:
                learning_rate = self.max_alpha_move / abs(gr.grad_alpha)
                print(f"iter: {iter}, alpha move too big: {alpha_move} due to grad_alpha {gr.grad_alpha:.4f}, lowered learning_rate: {learning_rate}")
                gr['learning_rate'] = learning_rate
                gr['msg']  = 'alpha move too big'
                self.append_descent_history(gr)

            # calculate mllk of a possible next move
            mllk1 = self.make_next_move(gr, learning_rate)
            gr['mllk'] = mllk1
            gr['kurt'] = self.lkc.rv_kurtosis
            gr['skew'] = self.lkc.rv_skew
            grad = _gr2grad(gr, round=6).to_dict()
            print(f"iter: {iter}, mllk?: {mllk1:.8f}, grad: {grad}, learning_rate: {learning_rate:.6f}")

            if abs(mllk1) < 1e-7:
                print(f"iter: {iter}, mllk*: {mllk1:.8f} vanished, stop")  # somehow SPX has this issue, grad_alpha jumps suddenly
                gr['msg']  = 'stop, mllk valished'
                self.append_descent_history(gr)
                self.lkc.copy_rv_to(self.lkc2)  # sync both
                break

            if mllk1 < mllk:
                # accept the move
                mllk = mllk1
                gr = _calc_gradient(iter)
                learning_rate *= ( 1.0 + self.learning_bump ) 
                if learning_rate >= self.max_learning_rate:  
                    learning_rate = self.max_learning_rate
                    print(f"iter: {iter}, capped learning_rate: {learning_rate}")

                gr['learning_rate'] = learning_rate
                gr['kurt'] = self.lkc.rv_kurtosis
                gr['skew'] = self.lkc.rv_skew
                gr['msg']  = 'accepted'
                self.append_descent_history(gr)
                grad = _gr2grad(gr, round=6).to_dict()
                self.lkc.calc_mllk_with_regularization(debug=True, iter=iter)  # just to print a message for regularization
                print(f"iter: {iter}, mllk+: {mllk:.8f} accepted, skew: {gr['skew']:.4f}, kurt: {gr['kurt']:.2f}, grad: {grad}")
            else:
                # reject the move
                self.lkc2.copy_rv_to(self.lkc)
                # reduce learning rate and try to make a move again
                if iter < 200:
                    learning_rate /= ( 1.0 + self.learning_bump/2 )
                else:
                    learning_rate /= ( 1.0 + self.learning_bump )  # retreat faster
                print(f"iter: {iter}, mllk-: {mllk1:.8f} not decreasing, vs prev {mllk}, lower learning_rate: {learning_rate}")

                gr['learning_rate'] = learning_rate
                if learning_rate < self.min_learning_rate: 
                    print(f"iter: {iter}, diminished learning_rate: {learning_rate}, stop")
                    gr['msg']  = 'stop, diminished learning_rate'
                    self.append_descent_history(gr)
                    self.lkc.copy_rv_to(self.lkc2)  # sync both
                    break
                else:
                    gr['msg']  = 'rejected, mllk not decreasing'
                    self.append_descent_history(gr)

            iter += 1
        
        if iter >= self.max_iter:
            msg = f"WARN: max iteration reached: iter {iter} vs max {self.max_iter}"
            if accept_max_iter:
                print(msg)
                gr['msg']  = 'stop, max iteration reached'
                self.append_descent_history(gr)
            else:
                raise ValueError(msg)
        print(f"end time: {datetime.now()}")
        return gr


class FCM_Px_Plot:
    # This class is catered to plotting FCM LME results for VIX Daily PX data
    # It uses LLK_Calculator as the data source, which accept FCM, FCM Inverse, and FCM Characteristic
    # VIX distribution is an important demo for the FCM use case
    # But it is flexible enough to be used for other data as well

    label_dict = {
        'inv': 'Inverse',
        'char': 'Characteristic',
    }

    def __init__(self, lkc: LLK_Calculator, title_prefix, xlabel='Px'):
        self.lkc = lkc
        self.title_prefix = title_prefix
        self.xlabel = xlabel + f' (rescaled by 1/{self.lkc.raw_scale:.2f})'

        self.fcm_label = self.get_fcm_label()
        self.rv_df = self.get_rv_df()

    def get_fcm_label(self) -> str:
        fcm_label = 'FCM'
        fcm_label += (' ' + self.label_dict[self.lkc.use_inverse]) if self.lkc.use_inverse is not None else ''  
        return fcm_label

    def get_rv_df(self) -> pd.DataFrame:
        max_x = self.lkc.x.max() * 1.2
        x0 = np.linspace(0.1, max_x, 200)
        df1 = pd.DataFrame({'x': x0})
        df1['pdf'] = df1['x'].apply(lambda x: self.lkc.rv.pdf(x))  # type: ignore
        df1['cdf'] = df1['x'].apply(lambda x: self.lkc.rv.cdf(x))  # type: ignore
        df1['ccdf'] = 1.0 - df1['cdf']
        return df1

    def plot_hist(self, ax, cumulative: Union[bool, Literal[1, -1]] = False, bins=200):
        return ax.hist(
            self.lkc.x, bins=bins, density=True, alpha=0.6, color='g', 
            cumulative=cumulative,
            label='Data histogram')

    def plot_pdf(self, ax, use_pdf=True, log_scale=False):
        lkc = self.lkc
        assert lkc.rv is not None

        if use_pdf:
            ax.plot(self.rv_df['x'], self.rv_df['pdf'], linewidth=1.5, color='red', label=self.fcm_label + ' pdf')
            ylabel = 'Density'
        else:
            ax.plot(self.rv_df['x'], self.rv_df['ccdf'], linewidth=1, color='red', label=self.fcm_label + ' ccdf')
            ylabel = 'Complimentary Cuml Density'

        if log_scale:
            ax.set_yscale('log')
            ylabel += ' (log)'

        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(ylabel)
        ax.axvline(x=lkc.loc, color='red', linestyle='--', label='fcm loc')
        ax.axvline(x=lkc.rv.mean(), color='orange', linestyle='--', label='fcm mean')
        
        if use_pdf:
            ax.axvline(x=lkc.x_percentile_20, color='orange', linestyle='--', linewidth=0.5, label=f'{self.fcm_label} 20th pctile')
            ax.axvline(x=lkc.x_percentile_66, color='orange', linestyle='--', linewidth=0.5, label=f'{self.fcm_label} 66th pctile')
        if not use_pdf:
            ax.axhline(y=1 - 0.20, color='orange', linestyle='--', linewidth=0.5, label=f'{self.fcm_label} 20th pctile')
            ax.axhline(y=1 - 0.66, color='orange', linestyle='--', linewidth=0.5, label=f'{self.fcm_label} 66th pctile')
            if log_scale:
                ax.axhline(y=1 - 0.99, color='orange', linestyle='--', linewidth=0.5, label=f'{self.fcm_label} 99th pctile')
                ax.axhline(y=1 - 0.997, color='orange', linestyle='--', linewidth=0.5, label=f'{self.fcm_label} 99.7th pctile')
        
        ax.legend()

    def plot_4_panels(self, ax1, ax2, ax3, ax4):
        lkc = self.lkc

        count, bins, ignored = self.plot_hist(ax1)
        delta = bins[1] - bins[0]
        bin_centers = (bins[:-1] + bins[1:]) / 2
        df2 = pd.DataFrame({'x': bin_centers, 'pdf': count * delta})
        df2['cdf'] = df2.pdf.cumsum()

        self.plot_pdf(ax1)
        ax1.set_title(f'{self.title_prefix} - {self.fcm_label} Fit')
        ax1.set_xlim([0, lkc.x.max() * 0.7])

        self.plot_hist(ax2)
        self.plot_pdf(ax2, log_scale=True)
        ax2.set_ylim([1e-3, 1.0])
        ax2.set_title(f'{self.fcm_label}: alpha={lkc.alpha:.3f}, k={lkc.k:.2f}, scale={lkc.scale:.2f}, loc={lkc.loc:.2f}')

        self.plot_hist(ax3, cumulative=-1)
        self.plot_pdf(ax3, use_pdf=False)
        ax3.set_xlim([0, lkc.x.max() * 0.7])

        self.plot_hist(ax4, cumulative=-1)
        self.plot_pdf(ax4, use_pdf=False, log_scale=True)
        ax4.set_ylim([1e-4, 1.5])