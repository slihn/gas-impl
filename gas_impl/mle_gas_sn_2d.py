# One-dimensional MLE for the gas-sn model

from typing import Optional, Dict, Union
import pandas as pd
import numpy as np 
from datetime import datetime
import json

from numpy import histogram, histogram2d  # type: ignore
from random import sample
from scipy.stats import skew, kurtosis

from .gas_sn_dist import gas_sn
from .multivariate_sn import Multivariate_GAS_SN, Multivariate_GAS_SN_Adp_2D, Cannonical_GAS_SN_Transform
from .utils import nearestPD
from .ff_dist import frac_f, FracF_PPF, Frac_F_Std_Adp_2D
from .mle_gas_sn import LLK_Calculator


# 2D Ellipitical 
def theta2rho(rhotheta):
    return np.arctan(rhotheta) / (np.pi/2)  # type: ignore

def rho2theta(rho):
    return np.tan(rho * np.pi/2)  # type: ignore


class LLK_Calculator_2D:
    def __init__(self, data: pd.DataFrame, compress_ratio=10.0, min_bins=50, is_adp=False, quadratic_mllk=False):
        self.raw_data = data
        assert isinstance(data, pd.DataFrame)
        assert 'x' in self.raw_data.columns
        assert 'y' in self.raw_data.columns
        self.data: pd.DataFrame = self.raw_data[['x', 'y']].copy()  # type: ignore
        # standardize the data to 1-std for MLE
        for c in ['x', 'y']:
            self.data[c] = self.data[c] / self.data[c].std()
        self.empirical_corr = self.data.x.corr(self.data.y)
 
        self.compress_ratio: float = float(compress_ratio)
        self.min_bins: int = int(min_bins)
        self.target_bins: int = self.get_target_bins()
        self.is_adp: bool = bool(is_adp)
        self.quadratic_mllk = quadratic_mllk  # TODO not sure how to implement this yet
        print(f"target_bins = {self.target_bins}, compress_ratio = {self.compress_ratio}")

        # unrealistic initial values as placeholders
        self.hyper_params: pd.Series = pd.Series()
        self.rv: Union[Multivariate_GAS_SN, Multivariate_GAS_SN_Adp_2D]
        self._minus_log_likehood: float = np.nan
        self._minus_log_likehood_ds: pd.Series = pd.Series()

        self.delta4 = 1e-4
        self.delta5 = 1e-5
        self.squared_lambda = 0.0  # user must set it to non-zero to use it, this is a tricky feature
        self.squared_cdf_lambda = 0.0  # user must set it to non-zero to use it, this is a tricky feature
        self.marginal_cdf_lambda = 0.0
        self.marginal_std_lambda = 0.0  # user must set it to non-zero to use it, this is a tricky feature

    def clear_mllk_data(self):
        # this is used to clear the MLLK data when you want to re-calculate MLLK
        self._minus_log_likehood = np.nan
        self._minus_log_likehood_ds: pd.Series = pd.Series()
        return self
    
    def set_mllk_data(self, key: str, value: float):
        self._minus_log_likehood_ds[key] = value
        return self
    
    @property
    def param_config(self) -> pd.DataFrame:
        if not self.is_adp:
            ls = [
                {'name': 'alpha', 'delta': self.delta4, 'method': 'mul'},
                {'name': 'k',     'delta': self.delta4, 'method': 'mul'},
            ]
        else:
            ls = [
                {'name': 'alpha0', 'delta': self.delta4, 'method': 'mul'},
                {'name': 'alpha1', 'delta': self.delta4, 'method': 'mul'},
                {'name': 'k0',     'delta': self.delta4, 'method': 'mul'},
                {'name': 'k1',     'delta': self.delta4, 'method': 'mul'},
                {'name': 'mult2',  'delta': self.delta4, 'method': 'mul'},  # 1.105  # TODO this is a hack, I can't explain why this is needed

            ]
        
        ls.extend([
            {'name': 'w0',    'delta': self.delta4, 'method': 'mul'},
            {'name': 'w1',    'delta': self.delta4, 'method': 'mul'},
            {'name': 'rhotheta', 'delta': self.delta4, 'method': 'delta'},  # use tan/arctan to keep it in range
            {'name': 'beta0', 'delta': self.delta4, 'method': 'delta'},
            {'name': 'beta1', 'delta': self.delta4, 'method': 'delta'},
            {'name': 'loc0',  'delta': self.delta5, 'method': 'delta'},
            {'name': 'loc1',  'delta': self.delta5, 'method': 'delta'},
        ])
        df = pd.DataFrame(ls)
        assert df.set_index('name').index.is_unique
        return df

    def clone(self):
        mle =  LLK_Calculator_2D(self.data, compress_ratio=self.compress_ratio, min_bins=self.min_bins)
        self.copy_rv_to(mle)
        return mle

    def set_compress_ratio(self, compress_ratio):
        # this should handle the side effect risen from a new compress_ratio
        self.compress_ratio = float(compress_ratio)
        self.target_bins = self.get_target_bins()
        self.clear_mllk_data()
        return self
    
    def get_marginal_1d_lkc(self, i):
        assert i in range(self.rv.n)
        marginal = self.rv.marginal_1d_rv(i)
        col = 'x' if i == 0 else 'y'
        lkc1 = LLK_Calculator(self.data[col], compress_ratio=self.compress_ratio, min_bins=self.target_bins)
        lkc1.init_rv_from_gas_sn_class(marginal)
        return lkc1

    def get_histogram2d(self, num_bins, positive_cnt=True) -> pd.DataFrame:
        counts, xedges, yedges = histogram2d(self.data['x'], self.data['y'], bins=(num_bins, num_bins))

        def _make_row(i,j):
            x = (xedges[i] + xedges[i+1]) / 2.0
            y = (yedges[j] + yedges[j+1]) / 2.0
            return {'x': x, 'y': y, 'cnt': counts[i,j]}

        df = pd.DataFrame([_make_row(i,j) for i in range(num_bins) for j in range(num_bins)])
        df = df.sort_values(by=['cnt'], ascending=False)
        if positive_cnt:
            df = df.query("cnt > 0").copy()
        return df 

    def get_target_bins(self):
        # find the number of bins that satisfies the compress_ratio
        ratio = 1.0
        bins = self.min_bins
        while bins < len(self.data):
            df2 = self.get_histogram2d(num_bins=bins)
            ratio = len(self.data) / len(df2)
            if ratio <= self.compress_ratio: break
            bins += 1
        return bins

    def get_target_histogram2d(self) -> pd.DataFrame:
        return self.get_histogram2d(self.target_bins, positive_cnt=True)

    def init_rv(self, hyper_params: pd.Series) -> 'LLK_Calculator_2D':
        self.hyper_params = s = hyper_params 
        beta = np.array([ s.beta0, s.beta1 ])
        loc = np.array([ s.loc0, s.loc1 ])
        w_arr = np.array([ s.w0, s.w1 ])
        # --------------------------------------------
        w = np.diag(w_arr)

        rho = theta2rho(s.rhotheta)  # type: ignore
        assert -1.0 <= rho <= 1.0, f"ERROR: rho is out of range: {rho} from {s.rhotheta}"
        corr = np.array([[ 1.0, rho ], [rho ,  1.0]])  # type: ignore
        cov = nearestPD(w @ corr @ w)

        if self.is_adp:
            alpha = np.array([ s.alpha0, s.alpha1 ])
            k = np.array([ s.k0, s.k1 ])
            self.rv = Multivariate_GAS_SN_Adp_2D(cov=cov, alpha=alpha, k=k, beta=beta, loc=loc)
        else:
            self.rv = Multivariate_GAS_SN(cov=cov, alpha=s.alpha, k=s.k, beta=beta, loc=loc)
        self.clear_mllk_data()
        return self
    
    def add_rv(self, lkc: 'LLK_Calculator_2D', d_params: pd.Series) -> 'LLK_Calculator_2D':
        s = lkc.hyper_params.copy()
        for key, val in d_params.items(): s[key] = s[key] + val  # type: ignore
        self.init_rv(s)
        return self

    def copy_rv_to(self, mle):
        mle.hyper_params = self.hyper_params.copy()
        mle.rv = self.rv
        mle.clear_mllk_data()
        return mle
    
    def get_hyper_params(self) -> pd.Series:
        return self.hyper_params[self.param_config['name']]

    def dump_hyper_params(self) -> str:
        parsed_json = json.loads(self.get_hyper_params().to_json())  # Convert to dict
        return json.dumps(parsed_json, indent=4, separators=(", ", ":\t"))  # Extra spaces after colon

    def get_quadratic_form(self, ascending=True, multiplier=1.0):
        data = np.array([ [r.x, r.y] for _, r in self.data.iterrows() ]) * multiplier
        assert isinstance(self.rv.cov, np.ndarray)
        Q = self.rv.quadratic_form(data)  # this is sorted
        if not ascending: Q = Q[::-1] 
        return Q
    
    def get_squared_rv(self):
        if not self.is_adp:
            return frac_f(self.rv.alpha, d=self.rv.n*1.0, k=self.rv.k)
        else:
            return self.rv._quadratic_rv()
    
    def get_squared_ppf_rv(self, multiplier=1.0) -> FracF_PPF:
        # this is used in plotting, when you call this, you freeze the quadratic form for the current self.rv in the return class
        print(f"setting up FracF_PPF with alpha: {self.rv.alpha}, k: {self.rv.k}")
        if not self.is_adp:
            assert isinstance(self.rv, Multivariate_GAS_SN)
            ff_rv = FracF_PPF(self.rv.alpha, d=self.rv.n, k=self.rv.k)
        else:
            assert isinstance(self.rv, Multivariate_GAS_SN_Adp_2D)
            # mean = np.mean(self.get_quadratic_form())  # not easy to calculate mean, just get it from the data
            # sd = mean / 2.0
            # rv = FracF_PPF(self.rv.alpha, d=self.rv.n, k=self.rv.k, rho=self.rv.rho,
            #                mean_override=mean, sd_override=sd, interp_size=4001)
            rv2 = self.rv._quadratic_rv()
            ff_rv = FracF_PPF(self.rv.alpha, d=self.rv.n, k=self.rv.k, rho=self.rv.rho, RV_override=rv2, sd_override=np.nan)
            
        ff_rv.set_observed_data(self.get_quadratic_form(multiplier=multiplier)) 
        return ff_rv  
   
    def calc_minus_log_likehood_from_rv(self, rv, df) -> float:
        # this MLLK is sum of log-likelihood, divided by the number of samples
        assert rv is not None
        df = df.copy()
        num_cols = len([c for c in df.columns if c in ['x', 'y']])
        def _log_pdf(r): 
            fn = rv.pdf
            if num_cols == 2:
                return np.log(fn([r.x, r.y]))  # type: ignore
            elif num_cols == 1:
                return np.log(fn(r.iloc[0]))
            else:
                raise ValueError(f"unknown number of columns: {num_cols}")

        def _apply_log_pdf(df) -> pd.Series:
            if self.is_adp and num_cols == 2:
                return df.apply(_log_pdf, axis=1)
            else:
                return df.parallel_apply(_log_pdf, axis=1)
            
        df['logpdf'] = _apply_log_pdf(df)
        mllk = df.eval("cnt * logpdf").sum() * -1.0 / df.cnt.sum()
        assert abs(mllk) >= 0, f"ERROR: mllk is nan: {mllk}"
        return mllk
 
    def calc_minus_log_likehood(self, use_hist=True, force=False, debug=False) -> float:
        # this MLLK is sum of log-likelihood, divided by the number of samples
        if abs(self._minus_log_likehood) >= 0 and not force: 
            return self._minus_log_likehood

        assert self.rv is not None            
        if not use_hist:
            df = self.data.copy()
            df['cnt'] = 1
        else:
            df = self.get_histogram2d(self.target_bins) 
        
        if not self.is_adp:
            # for ellipitical model, we can calculate regular MLLK
            mllk = self.calc_minus_log_likehood_from_rv(self.rv, df)
            self.set_mllk_data('mllk_ell', mllk)
            self._minus_log_likehood = mllk
            if debug:
                print(f"elliptic mllk: {mllk:.8f}")
            return self._minus_log_likehood
        else:
            # for adaptive model, too expensive to caluclate regular MLLK
            # we calculate two marginal MLLKs and combine them
            assert self.rv.n == 2
            mllks = np.zeros(self.rv.n)
            for i in range(self.rv.n):
                marginal_lkc = self.get_marginal_1d_lkc(i)
                mllks[i] = marginal_lkc.calc_mllk()
                self.set_mllk_data(f'mllk_{i}', mllks[i])
                if debug: print(f"marginal {i} mllk: {mllks[i]:.8f}")
            mllk = sum(mllks)
            self.set_mllk_data('mllk_adp', mllk)
            self._minus_log_likehood = mllk
            if debug:
                print(f"total marginal mllks: {self._minus_log_likehood}")

        assert abs(self._minus_log_likehood) >= 0, f"ERROR: mllk is nan: {self._minus_log_likehood}"
        return self._minus_log_likehood
    
    def calc_mllk(self, use_hist=True, force=False) -> float:
        # just a simpler name
        return self.calc_minus_log_likehood(use_hist=use_hist, force=force)

    def calc_mllk_squared(self, debug=False) -> float:
        mult2: float = self.hyper_params['mult2'] if self.is_adp else 1.0  # type: ignore
        m1_observed = np.mean(self.get_quadratic_form(multiplier=mult2))
        if not self.is_adp:
            m1_expected = self.get_squared_rv().moment(1)
        else:
            sz = 1000 * 1000 # this is pretty slow
            m1_expected = self.rv._quadratic_rvs_mean(sz)
        if debug: print(f"m1_observed: {m1_observed:.8f}, m1_expected: {m1_expected:.8f}")
        return (m1_observed - m1_expected)**2


    def calc_regularization(self, lambda2: float, lambda3: float, lambda4: float, debug=False) -> pd.Series:
        assert self.rv is not None
        obj = pd.Series({'total': 0.0})
        if lambda2 != 0:
            assert self.rv is not None
            theoretical_corr = self.rv.var2rho()
            obj['corr'] = lambda2 * (theoretical_corr - self.empirical_corr)**2  # type: ignore
            if debug: 
                print(f"theoretical_corr: {theoretical_corr:.4f}, empirical_corr: {self.empirical_corr:.4f}, obj_corr: {obj['corr']:.8f}")

        # mid point of CDF of the quadratic form
        if self.squared_cdf_lambda != 0:
            mult2: float = self.hyper_params['mult2'] if self.is_adp else 1.0  # type: ignore
            q_rv_mid = self.get_squared_rv().ppf(0.5)
            q_data_mid = np.median(self.get_quadratic_form(multiplier=mult2))
            obj['sqr_cdf'] = self.squared_cdf_lambda * (q_rv_mid - q_data_mid)**2  # type: ignore
            if debug: 
                print(f"q_rv_mid: {q_rv_mid:.4f}, q_data_mid: {q_data_mid:.4f}, obj_sqr_cdf: {obj['sqr_cdf']:.8f}")

        assert self.rv.n == 2
        for i in range(self.rv.n):
            marginal_lkc = self.get_marginal_1d_lkc(i)
            marginal = marginal_lkc.rv
            x: pd.Series = marginal_lkc.data

            obj[f'skew{i}'] = lambda3 * marginal_lkc.L2_skew()
            obj[f'kurtosis{i}'] = lambda4 * marginal_lkc.L2_kurtosis()
            
            if self.marginal_std_lambda != 0:
                x_rv_std   = marginal_lkc.rv_var**0.5
                x_data_std = marginal_lkc.x_scale  # type: ignore
                obj_value = self.marginal_std_lambda * marginal_lkc.L2_std()  # type: ignore
                obj[f'std{i}'] = obj_value
                if debug:
                    print(f"marginal {i}- rv_std: {x_rv_std:.4f}, x_std: {x_data_std:.4f}, obj_std: {obj_value:.8f}")

            if self.marginal_cdf_lambda != 0:
                x_rv_mid = marginal.ppf(0.5)  # type: ignore
                x_data_mid = np.median(np.sort(x))  # type: ignore
                obj_value = self.marginal_cdf_lambda * (x_rv_mid - x_data_mid)**2  # type: ignore
                obj[f'cdf{i}'] = obj_value
                if debug: 
                    print(f"marginal {i}- rv_mid: {x_rv_mid:.4f}, data_mid: {x_data_mid:.4f}, obj_cdf: {obj_value:.8f}")

            if self.marginal_cdf_lambda != 0:
                q_rv_mid = marginal_lkc.get_squared_rv().ppf(0.5)
                q_data_mid = np.median(marginal_lkc.get_squared_x())  # type: ignore
                obj_value = self.marginal_cdf_lambda * (q_rv_mid - q_data_mid)**2  # type: ignore
                obj[f'sqr_cdf{i}'] = obj_value
                if debug:
                    print(f"marginal {i} - Q's rv_mid: {q_rv_mid:.4f}, data_mid: {q_data_mid:.4f}, obj_sqr_cdf: {obj_value:.8f}")

        obj['total'] = obj.sum()
        return obj

    def calc_mllk_with_regularization(self, lambda2: float = 0.0, lambda3: float = 0.0, lambda4: float = 0.0, debug=False) -> float:
        A = self.calc_mllk()
        B = self.calc_mllk_squared() * self.squared_lambda if self.squared_lambda > 0 else 0.0
        reg = self.calc_regularization(lambda2, lambda3, lambda4, debug=debug)
        C = float(reg['total'])  # type: ignore
        if debug: print(f"mllk-1: {A:.8f}, mllk-2: {B:.8f}, reg: {C:.8f} from {reg.to_dict()}")
        assert abs(A) >= 0, f"ERROR: mllk A is nan: {A}"
        assert abs(B) >= 0, f"ERROR: mllk B is nan: {B}"
        assert abs(C) >= 0, f"ERROR: mllk C is nan: {C}"
        return float(A + B + C)

    def get_hyper_params_delta(self, r: pd.Series) -> pd.Series:
        d_params_dict: Dict = {}
        for _, c in self.param_config.iterrows():
            assert isinstance(c, pd.Series)
            if c['method'] == 'mul':
                d_params_dict[c['name']] = r[c['name']] * c['delta']  # type: ignore
            elif c['method'] == 'delta':
                d_params_dict[c['name']] = c['delta']
            else:
                raise ValueError(f"unknown method: {c['method']}")
        return pd.Series(d_params_dict)

    def increment_hyper_params(self, s: pd.Series) -> pd.Series:
        r = self.hyper_params.copy()
        return pd.Series({ 
            key: r[key] + s.get(key, 0.0) for key in self.param_config['name']  # type: ignore
        })

    def calc_gradient(self, lambda2: float = 0.0, lambda3: float = 0.0, lambda4: float = 0.0, debug=False):        
        r = self.hyper_params.copy()
        d_params = self.get_hyper_params_delta(r)
        # print(d_params)

        def _calc_mllk(d_params: pd.Series, debug=False) -> float:
            s = r.copy()
            if len(d_params) > 0:
                for key, val in d_params.items(): 
                    if key in s:  s[key] = s[key] + val  # type: ignore
            self.init_rv(s)  # type: ignore
            mllk = self.calc_mllk_with_regularization(lambda2, lambda3, lambda4, debug=debug)
            return mllk

        r['mllk'] = _calc_mllk(pd.Series(), debug=(True if debug else False))
        reg = self.calc_regularization(lambda2, lambda3, lambda4)
        for key, val in reg.items():  r[f"mllk_reg_{key}"] = val
        r['mllk_reg'] = reg['total']
        for key, val in d_params.items():
            assert isinstance(key, str)
            assert isinstance(val, float)
            r['d_' + key] = val
            r['grad_' + key] = (_calc_mllk(pd.Series({key: val})) - r.mllk) / val
            # if debug:  print(f"key: {key}, grad: {r['grad_' + key]:.6f}, at {r[key]:.6f}")

        if debug:
            tm = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"mllk: {r.mllk:.6f}, mllk_reg: {r.mllk_reg:.6f}, tm: {tm}")

        # ensure calc_gradient() does not change the RV state of the object at the end
        self.init_rv(r)  # type: ignore
        return r


class MLE_2D:
    def __init__(self, lkc: LLK_Calculator_2D, 
                 max_iter: int = 200  # show
                 ):
        self.lkc = lkc
        self.lkc2 = lkc.clone()  # as a storage place during the gradient descent phase
        
        # for gradient descent
        self.learning_rate = 0.5  # initial learning rate
        self.min_learning_rate = 1e-3 # stop when it's too small
        self.max_learning_rate = 2.0  # cap it so it won't jump randomly
        self.learning_bump = 0.4
        self.max_iter = max_iter  # 40 minutes
        self.max_alpha_move = 0.02  # don't allow alpha to jump too big

        # results are stored below
        self.scan_result = pd.DataFrame()  # stores the result of scan_param_space()
        self.descent_history = pd.DataFrame()  # stores the result of gradient descent

    # --------------------------------------------------------------------------------
    def make_next_move(self, gr: pd.Series, learning_rate: float, lambda2: float, lambda3: float, lambda4: float) -> float:
        next_move = gr.filter(like='grad_')
        next_move.index = next_move.index.str.replace('grad_', '') 
        next_move = next_move.apply(lambda x: x * (np.random.rand() * 0.5 + 0.5) * learning_rate * -1.0)
        print(f"next move: {next_move.to_dict()}")
        self.lkc.copy_rv_to(self.lkc2)
        self.lkc.init_rv(self.lkc.increment_hyper_params(next_move))
        return self.lkc.calc_mllk_with_regularization(lambda2, lambda3, lambda4)
    
    def append_descent_history(self, gr: pd.Series) -> pd.DataFrame:
        gr['timestamp'] = datetime.now()
        df = pd.DataFrame(data= [gr])
        if len(self.descent_history) == 0:
            self.descent_history = df
        else:
            self.descent_history = pd.concat([self.descent_history, df])
        return self.descent_history

    def param_msg(self, gr: pd.Series) -> str:
        if self.lkc.is_adp:
            alpha_k_msg = f"alpha 0/1: {gr.alpha0:.6f} / {gr.alpha1:.6f}, k: {gr.k0:.6f} / {gr.k1:.6f}"
        else:
            alpha_k_msg = f"alpha: {gr.alpha:.6f}, k: {gr.k:.6f}, mult2: {gr.mult2:.6f}"
        return f"{alpha_k_msg}, rho2: {gr.rhotheta:.6f} / {theta2rho(gr.rhotheta):.4f}, " + \
               f"beta: {gr.beta0:.6f} / {gr.beta1:.6f}, scale: {gr.w0:.6f} / {gr.w1:.6f}, loc: {gr.loc0:.6f} / {gr.loc1:.6f}"

    def gradient_descent(self, lambda2: float = 0.0, lambda3: float = 0.0, lambda4: float = 0.0, accept_max_iter=True) -> pd.Series:
        self.lkc.copy_rv_to(self.lkc2)
        start_time = datetime.now()
        print(f"lkc compress ratio: {self.lkc.compress_ratio}, bins: {self.lkc.target_bins}, max_iter: {self.max_iter}, " + 
              f"lambda2: {lambda2}, lambda3: {lambda3}, lambda4: {lambda4}, " +
              f"sqr m1: {self.lkc.squared_lambda}, sqr cdf: {self.lkc.squared_cdf_lambda}, " + 
              f"mg cdf: {self.lkc.marginal_cdf_lambda}, mg std: {self.lkc.marginal_std_lambda}")

        def _calc_gradient(iter: int) -> pd.Series:
            debug = True if self.lkc.squared_lambda > 0 else False  # we need to see the debug messagefor mllkc
            gr = self.lkc.calc_gradient(lambda2=lambda2, lambda3=lambda3, lambda4=lambda4, debug=debug)
            gr['iter'] = iter
            print(f"iter: {iter}, mllkc: {gr.mllk:.8f} / {gr.mllk_reg:.8f} from " + self.param_msg(gr))
            return gr
        
        def _gr2grad(gr: pd.Series, round=None) -> pd.Series:
            grad = gr[[x for x in gr.index.values if x.startswith('grad_')]].copy()
            if round is not None:
                grad = grad.apply(lambda x: np.round(x, decimals=round))  # type: ignore  # pandas series round has a bug !
            return grad

        iter = 1 
        gr = _calc_gradient(iter)
        learning_rate = self.learning_rate
        while iter <= self.max_iter:
            mllk = gr.mllk

            # don't allow alpha to jump too big
            grad_alpha  = max([gr.grad_alpha0, gr.grad_alpha1]) if self.lkc.is_adp else gr.grad_alpha
            alpha_move = grad_alpha * learning_rate
            if abs(alpha_move) > self.max_alpha_move:
                learning_rate = self.max_alpha_move / abs(grad_alpha)
                print(f"iter: {iter}, alpha move too big: {alpha_move} due to grad_alpha {grad_alpha:.4f}, lowered learning_rate: {learning_rate}")
                gr['learning_rate'] = learning_rate
                gr['msg']  = 'alpha move too big'
                self.append_descent_history(gr)

            # calculate mllk of a possible next move
            mllk1 = self.make_next_move(gr, learning_rate, lambda2, lambda3, lambda4)
            gr['mllk'] = mllk1
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
                gr['msg']  = 'accepted'
                self.append_descent_history(gr)
                grad = _gr2grad(gr, round=6).to_dict()
                elapsed_time = (datetime.now() - start_time).total_seconds()
                print(f"iter: {iter}, mllk+: {mllk:.8f} accepted, elapsed: {elapsed_time:.1f} sec, avg: {elapsed_time/iter:.1f} sec, grad: {grad}")

                # dump result after accepted move
                iter_mod = 5 if self.lkc.is_adp else 20
                if iter % iter_mod == 0:
                    dttm = datetime.now().strftime('%m%d_%H%M%S') + '_' + str(iter)
                    print(f"hyper_{dttm} = pd.Series(" + self.lkc.dump_hyper_params() + ")")
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

        elapsed_time = (datetime.now() - start_time).total_seconds()
        print(f"iter: {iter}, elapsed: {elapsed_time:.2f} seconds, avg: {elapsed_time/iter:.2f} seconds")
        return gr



