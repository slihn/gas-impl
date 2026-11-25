"""
GSaS_JumpRegimes: split a return series into N regimes with a statistical jump model, then fit GSaS to the returns
of each regime. The distribution of the ticker is the weighted mixture of the N regime GSaS distributions.

The regime labels come from the jump model of Shu, Yu and Mulvey (2024), as in the BMDA dual-regime code, through
the `jumpmodels` package. Features (for halflives 1, 5, 10, 21 days): the EWM return, the log EWM downside
deviation and their ratio (an EWM Sortino ratio); each is clipped to its mean +- 4 sd (clip_sd) and standardized.
Clipping only bounds the jump model's input: every day keeps its label and the GSaS fits use all returns. The share
of feature values and of days clipped is reported by clip_report() (about 0.3% of values and 2% of days at 4 sd).

How the regimes, and so the weights, are determined. The jump model chooses a regime s_t for every day and a
feature centroid theta_j for every regime to minimize

    sum_t 1/2 |y_t - theta_{s_t}|^2  +  lambda * #{t : s_t != s_{t-1}},

the squared distance of each day's features y_t from its regime centroid, plus the jump penalty lambda for every
regime switch. It alternates two exact steps until the objective stops falling: given the centroids, the best
regime sequence by dynamic programming (Viterbi); given the sequence, each centroid is the mean of its regime's
features. It runs from n_init k-means++ starts and keeps the lowest objective. lambda = 0 is k-means on the
features (no persistence); a larger lambda gives fewer, longer regimes.

The weight of regime j is then its share of days, w_j = #{t : s_t = j} / n, and the regime's GSaS is GSaS_Fit on
its returns (the moment-and-density fit). The mixture density is sum_j w_j f_j. A regime whose returns cannot be
fitted is drawn as a normal with its sample mean and sd: fewer than 30 days, excess kurtosis <= 0, or a shape
outside the GSaS region (e.g. a flatter peak than the normal's with positive kurtosis, which happens when a regime
mixes periods of different drift: the features include EWM returns, so regimes separate by trend as well as
volatility).

Regimes are numbered 1..N from calm to turbulent (by the sd of their returns).

Usage:
    reg = GSaS_JumpRegimes(xs, n_regimes=2, jump_penalty=50).fit()   # xs: Series of simple returns, dated
    reg.summary()             # DataFrame, one row per regime
    reg.labels                # Series of regime numbers 1..N on the dates of xs
    reg.transition_matrix()   # DataFrame (N, N), the observed day-to-day switching frequencies
    reg.clip_report()         # DataFrame, the share of days each feature is clipped, plus rows 'any feature', 'all values'
    reg.pdf(xg)               # ndarray, the mixture density
    reg.component_pdfs(xg)    # ndarray (len(xg), N), w_j f_j
"""
import numpy as np
import pandas as pd
from jumpmodels.jump import JumpModel
from scipy.stats import norm

from .gsas_fit import GSaS_Fit


class GSaS_JumpRegimes:
    HALFLIVES = (1, 5, 10, 21)

    def __init__(self, xs, n_regimes=2, jump_penalty=50.0, clip_sd=4.0, n_init=10, random_state=0):
        """
        xs: Series of simple returns indexed by date (an array is given a 0..n-1 index). n_regimes: N.
        jump_penalty: lambda, the cost of one regime switch in units of the standardized squared feature distance.
        clip_sd: features are clipped to mean +- clip_sd sd before scaling. n_init, random_state: jump model starts.
        """
        xs = pd.Series(xs, dtype=float) if not isinstance(xs, pd.Series) else xs.astype(float)
        if xs.isna().any() or len(xs) < 30*n_regimes:
            raise ValueError("xs must have no NaN and at least 30 returns per regime")
        self.xs = xs
        self.n_regimes = int(n_regimes)
        self.jump_penalty, self.clip_sd = float(jump_penalty), float(clip_sd)
        self.n_init, self.random_state = int(n_init), random_state
        self.fitted = False

    # ----------------------------------------------------------------------------------------------------
    # features and labels
    # ----------------------------------------------------------------------------------------------------
    @classmethod
    def raw_features(cls, xs):
        """EWM return, log EWM downside deviation and EWM Sortino per halflife. Returns a DataFrame like xs."""
        feats = {}
        for hl in cls.HALFLIVES:
            ret = xs.ewm(halflife=hl).mean()
            dd = np.sqrt(np.minimum(xs, 0.0).pow(2).ewm(halflife=hl).mean())
            feats[f'ret_{hl}'] = ret
            feats[f'DD-log_{hl}'] = np.log(dd)
            feats[f'sortino_{hl}'] = ret/dd
        return pd.DataFrame(feats)

    def _clean_features(self):
        # raw features with inf -> previous value, else 0; and the clip bounds mean -+ clip_sd sd
        f = self.raw_features(self.xs).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
        m, s = f.mean(), f.std(ddof=0)
        return f, m - self.clip_sd*s, m + self.clip_sd*s

    def features(self):
        """The jump model input: raw features cleaned (inf -> previous value, else 0), clipped and standardized."""
        f, lo, hi = self._clean_features()
        f = f.clip(lower=lo, upper=hi, axis=1)
        return (f - f.mean())/f.std(ddof=0).replace(0, 1)

    def clip_report(self):
        """
        How much the clip touches. Returns a DataFrame with columns low, high and total (clipped from below, from
        above, either) and one row per feature (the share of days that feature is clipped), then two summary rows:
        'any feature', the share of days with at least one feature clipped, and 'all values', the share of all
        feature values clipped.
        """
        f, lo, hi = self._clean_features()
        below, above = f.lt(lo, axis=1), f.gt(hi, axis=1)
        out = pd.DataFrame({'low': below.mean(), 'high': above.mean(), 'total': (below | above).mean()})
        out.loc['any feature'] = [below.any(axis=1).mean(), above.any(axis=1).mean(), (below | above).any(axis=1).mean()]
        out.loc['all values'] = [below.values.mean(), above.values.mean(), (below | above).values.mean()]
        return out

    def fit(self):
        """Label the days with the jump model and fit GSaS per regime. Returns self."""
        jm = JumpModel(n_components=self.n_regimes, jump_penalty=self.jump_penalty, cont=False,
                       n_init=self.n_init, random_state=self.random_state)
        jm.fit(self.features(), ret_ser=self.xs, sort_by='vol')
        raw = np.asarray(jm.labels_)
        order = np.argsort([self.xs.values[raw == j].std() if (raw == j).any() else np.inf
                            for j in range(self.n_regimes)])                       # calm to turbulent
        relabel = np.empty(self.n_regimes, dtype=int)
        relabel[order] = np.arange(1, self.n_regimes + 1)
        self.labels = pd.Series(relabel[raw], index=self.xs.index, name='regime')
        self.weights = np.array([(self.labels == j).mean() for j in range(1, self.n_regimes + 1)])
        self.fits = [self._fit_regime(self.xs.values[self.labels.values == j]) for j in range(1, self.n_regimes + 1)]
        self.jump_model = jm
        self.fitted = True
        return self

    @staticmethod
    def _fit_regime(x):
        # GSaS_Fit on a regime's returns, None when it has fewer than 30 days
        return GSaS_Fit(x) if len(x) >= 30 else None

    def _is_gsas(self, j):
        # regime j (0-based) is drawn as its GSaS fit; otherwise as a normal
        return self.fits[j] is not None and self.fits[j].status == 'ok'

    # ----------------------------------------------------------------------------------------------------
    # results
    # ----------------------------------------------------------------------------------------------------
    def _require_fit(self):
        if not self.fitted:
            raise RuntimeError("call fit() first")

    def runs(self):
        """The regime runs: a DataFrame with columns regime, start, end, days, one row per uninterrupted run."""
        self._require_fit()
        lab = self.labels
        brk = (lab != lab.shift()).cumsum()
        g = lab.groupby(brk)
        return pd.DataFrame({'regime': g.first().values, 'start': g.apply(lambda s: s.index[0]).values,
                             'end': g.apply(lambda s: s.index[-1]).values, 'days': g.size().values})

    def summary(self):
        """
        One row per regime, calm to turbulent. Returns a DataFrame with columns weight (share of days), days, runs
        (number of runs), mean_run (average run length in days), mean, sd, skew, exkurt, SPD, v1, v2 (the sample
        shape), alpha, k, xi (nan when not fitted), status (GSaS_Fit's: 'ok', 'exkurt <= 0' or 'no solution', the
        last when the sample shape lies outside the GSaS region; or 'too few days') and drawn_as ('GSaS' or
        'normal').
        """
        self._require_fit()
        runs = self.runs()
        rows = []
        for j, (w, fit) in enumerate(zip(self.weights, self.fits), start=1):
            x = self.xs.values[self.labels.values == j]
            r = runs[runs.regime == j]
            row = dict(regime=j, weight=w, days=len(x), runs=len(r), mean_run=r.days.mean(), mean=x.mean(),
                       sd=x.std(), skew=pd.Series(x).skew())
            if fit is not None:
                s = fit.summary()
                row.update({c: s[c] for c in ['exkurt', 'SPD', 'v1', 'v2', 'alpha', 'k', 'xi', 'status']})
            else:
                row.update(status='too few days')
            row['drawn_as'] = 'GSaS' if self._is_gsas(j - 1) else 'normal'
            rows.append(row)
        return pd.DataFrame(rows).set_index('regime')

    def transition_matrix(self):
        """The observed P(regime j tomorrow | regime i today). Returns a DataFrame (N, N), regimes 1..N."""
        self._require_fit()
        lab = range(1, self.n_regimes + 1)
        counts = pd.crosstab(self.labels.values[:-1], self.labels.values[1:]).reindex(index=lab, columns=lab,
                                                                                    fill_value=0)
        out = counts.div(counts.sum(axis=1), axis=0)
        out.index.name, out.columns.name = 'from', 'to'
        return out

    def _regime_pdf(self, j, xg):
        if self._is_gsas(j):
            return self.fits[j].pdf(xg)
        x = self.xs.values[self.labels.values == j + 1]
        return norm.pdf(xg, x.mean(), x.std())

    def _regime_cdf(self, j, xg):
        if self._is_gsas(j):
            return self.fits[j].cdf(xg)
        x = self.xs.values[self.labels.values == j + 1]
        return norm.cdf(xg, x.mean(), x.std())

    def component_pdfs(self, xg):
        """The weighted regime densities w_j f_j at xg (an array). Returns an ndarray (len(xg), N)."""
        self._require_fit()
        xg = np.asarray(xg, dtype=float)
        return np.column_stack([w*self._regime_pdf(j, xg) for j, w in enumerate(self.weights)])

    def pdf(self, xg):
        """The mixture density sum_j w_j f_j at xg (an array). Returns an ndarray of len(xg)."""
        return self.component_pdfs(xg).sum(axis=1)

    def cdf(self, xg):
        """The mixture cdf at xg (an array). Returns an ndarray of len(xg)."""
        self._require_fit()
        xg = np.asarray(xg, dtype=float)
        return sum(w*self._regime_cdf(j, xg) for j, w in enumerate(self.weights))

    def qq(self, n_grid=4001):
        """
        The Q-Q pairs of the mixture at the plotting positions (i - 1/2)/n, the cdf inverted on a grid of n_grid
        points. Returns a tuple (q_fit, x_sorted) of two ndarrays of len(xs).
        """
        xs = np.sort(self.xs.values)
        p = (np.arange(1, len(xs) + 1) - 0.5)/len(xs)
        grid = np.linspace(xs[0] - 0.5*abs(xs[0]), xs[-1] + 0.5*abs(xs[-1]), n_grid)
        return np.interp(p, self.cdf(grid), grid), xs
