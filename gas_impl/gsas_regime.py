"""
GSaS_JumpRegimes: split a return series into N regimes with a statistical jump model, then fit GSaS (or LNEP, where
GSaS has no solution) to the returns of each regime. The distribution of the ticker is the weighted mixture of the N
regime distributions.

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

The weight of regime j is then its share of days, w_j = #{t : s_t = j} / n. The mixture density is sum_j w_j f_j.
The family of each regime is chosen by where its sample shape (v1, v2), the kurtosis and the standardized peak
density, lies on the plane:

    GSaS    GSaS_Fit, when its status is 'ok': v2 >= 0, every normal scale mixture. It matches (v1, v2) exactly,
            with the regime's sample mean and sd; shape (alpha, k), and xi.
    LNEP    LNEP_Fit (lnep_dist.py) by maximum likelihood, when GSaS has no solution and the regime's shape is
            consistent with the LNEP region: v2 < 0, between the EP curve and the line v2 = v1 - 0.088. Consistent
            means that at least lnep_min_p (5%) of n_boot block-bootstrap resamples of the regime (GSaS_Fit's
            bootstrap_shape, 20-day blocks) land inside the region; the share is p_lnep. The point estimate alone
            is not used: the region is a band of width 0.088 in v1 - v2, while the sampling sd of v1 - v2 is about
            0.02-0.03 at a few thousand days, so one day of data can move a regime across the line. Maximum
            likelihood, not the moments, because a moment match near the line forces alpha towards infinity (the
            uniform kernel). Shape (lnep_alpha, s): the exponential-power exponent and the sd of the log-volatility.
            It also covers some regimes with excess kurtosis <= 0.
    normal  the regime's sample mean and sd, when neither fits or the regime has fewer than 30 days. A regime with
            v2 < 0 beyond the LNEP band has a flat top with tails too heavy for any single scale law: it most
            likely hides a sharp sub-regime, so try more regimes or another lambda (the 'note' column says so).

A flatter peak than the normal's (v2 < 0) happens when a regime mixes periods of different drift: the features
include EWM returns, so regimes separate by trend as well as volatility. The two families agree on the LNN
diagonal (GSaS as k -> inf is LNEP at alpha = 2), but not across v2 = 0 for v1 > 0: just above it GSaS has
alpha -> 2, k -> 3 and a power tail of index about 4, while just below it LNEP has light, lognormal-type tails.

Regimes are numbered 1..N from calm to turbulent (by the sd of their returns).

Simulation. simulate() generates synthetic returns from the fitted model the way a hidden Markov model generates
data (as Mixture-VAE's generate_hmm_data does): a regime path from the Markov chain with the observed
transition_matrix(), its first day drawn from the regime weights, then each day's return drawn independently from
its regime's law (GSaS by GAS_SN with beta = 0, LNEP by LNEP_Fit.rvs_of, or the normal) with the regime's location
and sd. Within a regime the days are iid, so the persistence comes from the regime path alone.

Usage:
    reg = GSaS_JumpRegimes(xs, n_regimes=2, jump_penalty=50).fit()   # xs: Series of simple returns, dated
    reg.summary()             # DataFrame, one row per regime; drawn_as is 'GSaS', 'LNEP' or 'normal'
    reg.labels                # Series of regime numbers 1..N on the dates of xs
    reg.transition_matrix()   # DataFrame (N, N), the observed day-to-day switching frequencies
    reg.clip_report()         # DataFrame, the share of days each feature is clipped, plus rows 'any feature', 'all values'
    reg.pdf(xg)               # ndarray, the mixture density
    reg.component_pdfs(xg)    # ndarray (len(xg), N), w_j f_j
    reg.simulate(n, seed)     # DataFrame of n synthetic days (default 10 len(xs)): columns regime, x
"""
import numpy as np
import pandas as pd
from jumpmodels.jump import JumpModel
from scipy.stats import norm

from .gas_sn_dist import GAS_SN
from .gsas_fit import GSaS_Fit
from .lnep_dist import LNEP_Fit


class GSaS_JumpRegimes:
    HALFLIVES = (1, 5, 10, 21)

    def __init__(self, xs, n_regimes=2, jump_penalty=50.0, clip_sd=4.0, n_init=10, random_state=0,
                 n_boot=500, lnep_min_p=0.05):
        """
        xs: Series of simple returns indexed by date (an array is given a 0..n-1 index). n_regimes: N.
        jump_penalty: lambda, the cost of one regime switch in units of the standardized squared feature distance.
        clip_sd: features are clipped to mean +- clip_sd sd before scaling. n_init, random_state: jump model starts.
        n_boot, lnep_min_p: a regime GSaS cannot fit is drawn as LNEP when at least lnep_min_p of n_boot block-
        bootstrap resamples of its shape lie inside the LNEP region.
        """
        xs = pd.Series(xs, dtype=float) if not isinstance(xs, pd.Series) else xs.astype(float)
        if xs.isna().any() or len(xs) < 30*n_regimes:
            raise ValueError("xs must have no NaN and at least 30 returns per regime")
        self.xs = xs
        self.n_regimes = int(n_regimes)
        self.jump_penalty, self.clip_sd = float(jump_penalty), float(clip_sd)
        self.n_init, self.random_state = int(n_init), random_state
        self.n_boot, self.lnep_min_p = int(n_boot), float(lnep_min_p)
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
        self.p_lnep = [self._p_lnep(fit) for fit in self.fits]
        self.lnep_fits = [LNEP_Fit(self.xs.values[self.labels.values == j]) if p >= self.lnep_min_p else None
                          for j, p in enumerate(self.p_lnep, start=1)]
        self.jump_model = jm
        self.fitted = True
        return self

    @staticmethod
    def _fit_regime(x):
        # GSaS_Fit on a regime's returns, None when it has fewer than 30 days
        return GSaS_Fit(x) if len(x) >= 30 else None

    def _p_lnep(self, gsas_fit):
        # share of block-bootstrap resamples whose shape lies inside the LNEP region; nan when GSaS fits or the
        # regime has fewer than 30 days (LNEP is not tried)
        if gsas_fit is None or gsas_fit.status == 'ok':
            return np.nan
        b = gsas_fit.bootstrap_shape(B=self.n_boot, seed=self.random_state)
        return float(np.mean([LNEP_Fit.invert(v1, v2)[2] == 'ok' for v1, v2 in zip(b.v1, b.v2)]))

    def _family(self, j):
        # how regime j (0-based) is drawn: 'GSaS', 'LNEP' or 'normal'
        if self.fits[j] is not None and self.fits[j].status == 'ok':
            return 'GSaS'
        if self.lnep_fits[j] is not None and self.lnep_fits[j].status == 'ok':
            return 'LNEP'
        return 'normal'

    def _model(self, j):
        # the fit object regime j is drawn from, or None for the normal
        return {'GSaS': self.fits[j], 'LNEP': self.lnep_fits[j], 'normal': None}[self._family(j)]

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
        shape), alpha, k, xi (the GSaS shape, nan unless drawn as GSaS), status (GSaS_Fit's: 'ok', 'exkurt <= 0' or
        'no solution', the last when the sample shape lies outside the GSaS region; or 'too few days'), drawn_as
        ('GSaS', 'LNEP' or 'normal'), lnep_alpha, s (the LNEP maximum-likelihood shape, nan unless drawn as LNEP),
        lnep_status (LNEP_Fit's 'ok' or 'not converged'; 'p < lnep_min_p' when the shape is not consistent with the
        LNEP region; '' when not tried), p_lnep (the bootstrap share inside the LNEP region; nan when not tried) and
        note (why a regime is drawn as a normal; '' otherwise).
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
            lf, p, fam = self.lnep_fits[j - 1], self.p_lnep[j - 1], self._family(j - 1)
            ok = lf is not None and lf.status == 'ok'
            lnep_status = lf.status if lf is not None else ('' if np.isnan(p) else f'p < {self.lnep_min_p:g}')
            note = ''
            if fam == 'normal':
                note = ('too few days' if fit is None else
                        'v2 < 0 beyond the LNEP band: likely hides a sharp regime' if fit.shape['v2'] < 0 else
                        'outside both regions')
            row.update(drawn_as=fam, lnep_alpha=lf.alpha if ok else np.nan, s=lf.s if ok else np.nan,
                       lnep_status=lnep_status, p_lnep=p, note=note)
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
        m = self._model(j)
        if m is not None:
            return m.pdf(xg)
        x = self.xs.values[self.labels.values == j + 1]
        return norm.pdf(xg, x.mean(), x.std())

    def _regime_cdf(self, j, xg):
        m = self._model(j)
        if m is not None:
            return m.cdf(xg)
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

    # ----------------------------------------------------------------------------------------------------
    # simulation: the regime path from the Markov chain, the returns from the regime laws
    # ----------------------------------------------------------------------------------------------------
    def _regime_rvs(self, j, n, rng):
        # n draws of regime j (0-based) from the law it is drawn as, with the regime's location and sd
        if n == 0:
            return np.empty(0)
        fam = self._family(j)
        if fam == 'GSaS':                    # GSaS_Fit's law: mean + sd Z/V standardized; GAS_SN at beta 0 is that law
            f = self.fits[j]
            g = GAS_SN(f.alpha, f.k, 0.0)
            return f.mean + f.sd*g.rvs(n, random_state=rng)/np.sqrt(g.var_by_log_moment())
        if fam == 'LNEP':
            m = self.lnep_fits[j]
            return LNEP_Fit.rvs_of(n, m.alpha, m.s, m.loc, m.sd, rng=rng)
        x = self.xs.values[self.labels.values == j + 1]
        return rng.normal(x.mean(), x.std(), n)

    def simulate(self, n=None, seed=0):
        """
        Synthetic returns from the fitted regime model, generated as by a hidden Markov model: the regime of day 0
        is drawn from the regime weights, each next regime from the row of transition_matrix() for today's regime,
        and each day's return independently from its regime's law (drawn_as: GSaS, LNEP or normal) with the regime's
        location and sd. n: the number of days, default 10 len(xs). seed: an int or a np.random.Generator. Returns a
        DataFrame indexed 0..n-1 with columns regime (1..N) and x.
        """
        self._require_fit()
        n = 10*len(self.xs) if n is None else int(n)
        rng = np.random.default_rng(seed)
        P = np.nan_to_num(self.transition_matrix().values)    # a row is nan (0/0) for a regime with no exits
        unseen = P.sum(axis=1) == 0                      # (seen only on the last day): it stays
        P[unseen] = np.eye(self.n_regimes)[unseen]
        cum = np.cumsum(P, axis=1)
        u = rng.random(n)
        s = np.empty(n, dtype=int)
        s[0] = rng.choice(self.n_regimes, p=self.weights)
        for t in range(1, n):
            s[t] = min(int(np.searchsorted(cum[s[t - 1]], u[t], side='right')), self.n_regimes - 1)
        x = np.empty(n)
        for j in range(self.n_regimes):
            on = s == j
            x[on] = self._regime_rvs(j, int(on.sum()), rng)
        return pd.DataFrame({'regime': s + 1, 'x': x})
