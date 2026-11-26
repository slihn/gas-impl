"""
Given an array of simple daily returns from one security, determine its GSaS shape (alpha, k).

The fit is the method of moments and density used by the gsas_ticker_fit notebooks (V4-xi, V4-sp500-stocks):
match the excess kurtosis gamma2 (a sample moment) and the standardized peak density SPD (a bin-free density
estimate at the centre, widened to the sparsity maximum when the centre dips; see peak_report) by inverting the one-to-one map (alpha, k) -> (gamma2, SPD). On the implied log-variance
plane, v1 = log(1 + gamma2/3)/4 and v2 = 2 log(sqrt(2 pi) SPD)/3; the LNN is the diagonal v1 = v2, k > 0 lies
below it and k < 0 above it. xi = sign(k)/z, with z = k - 1 (k > 0) or |k| (k < 0), is k rewritten so that the
LNN is xi = 0.

Usage:
    f = GSaS_Fit(returns)          # simple returns as decimals (0.01 = 1%); a numpy array or a pandas Series
    f = GSaS_Fit(returns, winsor_pct=0.5)   # optional: winsorize 0.5% in each tail first (default 0, no cut)
    f.shape                        # dict: n, exkurt, SPD, v1, v2
    f.skew                         # sample skewness; GSaS is symmetric, so a large |skew| means a poor fit
    f.alpha, f.k, f.xi, f.status   # the fit; nan and a non-'ok' status when there is no solution
    f.summary()                    # pd.Series of all of the above
    f.bootstrap_shape()            # pd.DataFrame of B block-bootstrap rows: v1, v2, skew
    f.gof()                        # pd.DataFrame of the H(-3) and Geary checks
    GSaS_Fit.peak_report(returns)  # dict: the peak density, the central-dip test and, on a dip, the widened window
"""
import warnings
from functools import cached_property

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.optimize import brentq
from scipy.special import gammaln
from scipy.stats import kurtosis, norm, skew, t as student_t

from .fcm_dist import FracChiMean


class GSaS_Fit:
    SQ2PI = np.sqrt(2*np.pi)
    PEAK_HS = np.array([0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20])   # quantile half-widths for the peak density
    PEAK_WIDEN_HS = np.round(np.arange(0.04, 0.4001, 0.01), 2)       # half-widths searched when the centre dips
    PEAK_RISE_SE = 2.65    # sd of the relative sparsity rise times sqrt(n), simulated on flat-top laws (EP(4), LNEP)
    PEAK_DIP_Z = 1.645     # one-sided 5% for the central-dip test
    A3_NORMAL = 2*np.sqrt(2/np.pi)                                    # E|Z|^3 of a standard normal
    A1_NORMAL = np.sqrt(2/np.pi)                                      # E|Z| of a standard normal
    ALPHAS = np.concatenate([np.geomspace(0.01, 0.5, 400), np.linspace(0.5, 1.999, 800)])   # alpha grid of the k-curves

    def __init__(self, x, winsor_pct=0.0):
        """
        x: simple daily returns as decimals (0.01 = 1%), 1-D, all finite. A pandas Series is accepted; only its
        values are used. Prices are rejected: a median |x| above 0.5 is taken to be a price series.
        winsor_pct: percent winsorized in EACH tail, in [0, 50). 0 (the default) cuts nothing; 0.5 clips x to its
        0.5th and 99.5th percentiles. Everything after (skew, shape, fit, bootstrap, gof, qq) uses the winsorized
        self.x; the original is kept in self.x_raw. The bootstrap resamples the winsorized x, it does not re-clip.
        Winsorizing is intrusive: clipping lowers v1 (tails) but hardly moves v2 (peak), so every series drifts above
        the LNN toward k < 0 (0.1% per tail moves SPY from xi +0.06 to -0.10). Use bootstrap_shape() for uncertainty.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or len(x) < 30:
            raise ValueError(f"x must be a 1-D array of at least 30 returns, got shape {x.shape}")
        if not np.isfinite(x).all():
            raise ValueError(f"x has {np.sum(~np.isfinite(x))} non-finite values; drop them first")
        if np.median(np.abs(x)) > 0.5:
            raise ValueError(f"median |x| is {np.median(np.abs(x)):.3g}: x looks like prices, not simple returns")
        if not 0 <= winsor_pct < 50:
            raise ValueError(f"winsor_pct is percent per tail in [0, 50), got {winsor_pct}")
        self.x_raw, self.winsor_pct = x, winsor_pct
        if winsor_pct > 0:
            x = np.clip(x, *np.percentile(x, [winsor_pct, 100 - winsor_pct]))
        self.n_winsorized = int(np.sum(x != self.x_raw))    # values moved to a clip bound, both tails
        self.x = x
        self.skew = float(skew(x))     # sample skewness (scipy, biased); GSaS is symmetric, so a large |skew| means a poor fit
        self.shape = self.sample_shape(x)

    # ----------------------------------------------------------------------------------------------------
    # GSaS moment algebra (exact log-Gamma form)
    # ----------------------------------------------------------------------------------------------------
    @staticmethod
    def _A_lim(x, e):
        # A(x) = lnG(eps x) - lnG(x/2), with its limit log(alpha/2) = -log(2 eps) at x = 0
        x = np.asarray(x, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(x == 0, -np.log(2*e), gammaln(e*x) - gammaln(x/2))

    @classmethod
    def K_of(cls, alpha, k):
        """
        The log-moment function of the FCM behind GSaS(alpha, k), K(s) = A(z + zeta s) - A(z), with
        z = k - 1 (k > 0) or |k| (k < 0) and zeta = sign(k). Returns a function of s; alpha may be an array.
        """
        e = 1.0/alpha
        zeta, z = (1.0, k - 1.0) if k > 0 else (-1.0, -k)
        return lambda s: cls._A_lim(z + zeta*s, e) - cls._A_lim(z, e)

    @classmethod
    def v12(cls, alpha, k):
        """
        The implied log-variance plane coordinates of GSaS(alpha, k):
            v1 = log(1 + gamma2/3)/4    = (K(-4) - 2K(-2))/4
            v2 = 2 log(sqrt(2 pi) SPD)/3 = 2(K(1) + K(-2)/2)/3
        Returns a tuple (v1, v2): floats for scalar alpha, ndarrays for an array alpha.
        """
        K = cls.K_of(alpha, k)
        return (K(-4) - 2*K(-2))/4, 2*(K(1) + 0.5*K(-2))/3

    @staticmethod
    def xi_of_k(k):
        """xi = sign(k)/z, with z = k - 1 (k > 0) or |k| (k < 0); xi = 0 is the LNN (k -> +-inf)."""
        return np.sign(k)/(k - 1 if k > 0 else -k)

    @staticmethod
    def k_of_xi(xi):
        """The inverse of xi_of_k: k = 1 + 1/xi (xi > 0) or 1/xi (xi < 0)."""
        return 1 + 1/xi if xi > 0 else 1/xi

    # ----------------------------------------------------------------------------------------------------
    # sample estimators
    # ----------------------------------------------------------------------------------------------------
    @staticmethod
    def sparsity(x, hs):
        """Quantile sparsity 2h/(q(0.5+h) - q(0.5-h)) at each half-width h: the mean density over the central 2h of x."""
        hs = np.asarray(hs, dtype=float)
        q = np.quantile(x, np.concatenate([0.5 - hs, 0.5 + hs]))
        return 2*hs/(q[len(hs):] - q[:len(hs)])

    @classmethod
    def peak_report(cls, x):
        """
        The peak density of x and how it was found. A single central peak makes the sparsity fall as the window
        widens, so the O(h^2) fit over PEAK_HS is extrapolated to h = 0. A central dip (as in TLT 2022-24) makes it
        rise instead; then the window is widened until the sparsity peaks, and that maximum is the peak.

        The centrality test is on rise = b h_max^2/a, the relative change of the fit a + b h^2 from h = 0 to
        h_max = max(PEAK_HS): a dip when rise > PEAK_DIP_Z * PEAK_RISE_SE/sqrt(n), one-sided 5% against a flat top
        (zero curvature, the boundary between a peak and a dip). A normal sample has rise ~ -0.05 and switches
        about 2% of the time at n = 753. On a dip, the sparsity over PEAK_WIDEN_HS is smoothed by a quadratic in h
        and its maximum taken, which avoids the upward bias of the maximum of the raw, noisy values.

        Returns a dict with keys:
            peak        the peak density, in the units of 1/x
            extrap      the h -> 0 extrapolation a
            rise        the relative rise b h_max^2/a
            threshold   the dip threshold for this n
            dip         bool, rise > threshold
            h_peak      the half-width of the widened maximum (0 when there is no dip)
        """
        n, hs = len(x), cls.PEAK_HS
        a, b = np.linalg.lstsq(np.vstack([np.ones_like(hs), hs**2]).T, cls.sparsity(x, hs), rcond=None)[0]
        rise, thr = b*hs.max()**2/a, cls.PEAK_DIP_Z*cls.PEAK_RISE_SE/np.sqrt(n)
        out = dict(peak=float(a), extrap=float(a), rise=float(rise), threshold=float(thr), dip=bool(rise > thr), h_peak=0.0)
        if out['dip']:
            hg = cls.PEAK_WIDEN_HS
            hh = np.linspace(hg[0], hg[-1], 721)
            v = np.polyval(np.polyfit(hg, cls.sparsity(x, hg), 2), hh)
            out.update(peak=float(v.max()), h_peak=float(hh[v.argmax()]))
        return out

    @classmethod
    def peak_density(cls, x, hs=None):
        """
        The bin-free density at the centre, in the units of 1/x; see peak_report for the method. With hs given, the
        plain O(h^2) extrapolation over those half-widths, without the centrality test.
        """
        if hs is None:
            return cls.peak_report(x)['peak']
        hs = np.asarray(hs, dtype=float)
        return float(np.linalg.lstsq(np.vstack([np.ones_like(hs), hs**2]).T, cls.sparsity(x, hs), rcond=None)[0][0])

    @classmethod
    def sample_shape(cls, x):
        """
        The two shape statistics of x and their plane coordinates. Returns a dict with keys:
            n       int, number of returns
            exkurt  excess kurtosis gamma2 (scipy, biased)
            SPD     standardized peak density = peak_density(x) * sd(x), with sd at ddof = 0
            v1      log(1 + exkurt/3)/4, the log-vol variance implied by the kurtosis
            v2      2 log(sqrt(2 pi) SPD)/3, the log-vol variance implied by the peak
        """
        g2, spd = float(kurtosis(x)), cls.peak_density(x)*x.std()
        return dict(n=len(x), exkurt=g2, SPD=spd, v1=np.log1p(g2/3)/4, v2=2*np.log(cls.SQ2PI*spd)/3)

    # ----------------------------------------------------------------------------------------------------
    # inversion (gamma2, SPD) -> (alpha, k)
    # ----------------------------------------------------------------------------------------------------
    @classmethod
    def invert(cls, gamma2, spd, a_lo=1e-3, a_hi=1.999, k_max=1e5):
        """
        (gamma2, SPD) -> (alpha, k) on the whole domain where the kurtosis exists: k > 5 - alpha or
        |k| > max(0, 1 - alpha). Unique by the injectivity proof; the branch is k > 0 below the LNN line
        v2 = v1, k < 0 above it. Returns a tuple (alpha, k), or (None, None) when there is no solution.
        """
        v1t, v2t = np.log1p(gamma2/3)/4, 2*np.log(cls.SQ2PI*spd)/3

        def a_on(k):
            lo = max(a_lo, (5 - k if k > 0 else 1 - abs(k)) + 1e-9)
            if lo >= a_hi:
                return None
            f = lambda a: cls.v12(a, k)[0] - v1t
            flo, fhi = f(lo), f(a_hi)
            if not (np.isfinite(flo) and np.isfinite(fhi)) or flo*fhi > 0:
                return None
            return brentq(f, lo, a_hi, xtol=1e-13)

        def g(k):
            a = a_on(k)
            return (np.nan, None) if a is None else (cls.v12(a, k)[1] - v2t, a)

        ks = np.geomspace(3.0001, k_max, 600) if v2t < v1t else -np.geomspace(1e-3, k_max, 600)
        vals = [g(k)[0] for k in ks]
        for k0, k1, g0, g1 in zip(ks, ks[1:], vals, vals[1:]):
            if np.isfinite(g0) and np.isfinite(g1) and g0*g1 <= 0:
                k = brentq(lambda kk: g(kk)[0], k0, k1, xtol=1e-12)
                return g(k)[1], k
        return None, None

    @cached_property
    def _alpha_k(self):
        if self.shape['exkurt'] <= 0:
            return None, None
        return self.invert(self.shape['exkurt'], self.shape['SPD'])

    @property
    def status(self):
        """'ok', 'exkurt <= 0' (a normal mixture is leptokurtic) or 'no solution'."""
        if self.shape['exkurt'] <= 0:
            return 'exkurt <= 0'
        return 'ok' if self._alpha_k[0] is not None else 'no solution'

    @property
    def alpha(self):
        a = self._alpha_k[0]
        return np.nan if a is None else a

    @property
    def k(self):
        k = self._alpha_k[1]
        return np.nan if k is None else k

    @property
    def xi(self):
        return self.xi_of_k(self.k) if self.status == 'ok' else np.nan

    @property
    def mean(self):
        return float(self.x.mean())

    @property
    def sd(self):
        return float(self.x.std())

    def summary(self):
        """
        Returns a pd.Series with index n, exkurt, SPD, v1, v2 (from shape), skew, alpha, k, xi (nan when not fitted),
        winsor_pct and status. This is the per-security row of the sp500-stocks table.
        """
        return pd.Series(dict(**self.shape, skew=self.skew, alpha=self.alpha, k=self.k, xi=self.xi,
                              winsor_pct=self.winsor_pct, status=self.status))

    def _require_fit(self):
        if self.status != 'ok':
            raise ValueError(f"no GSaS fit: {self.status} (exkurt {self.shape['exkurt']:.3f}, SPD {self.shape['SPD']:.4f})")

    # ----------------------------------------------------------------------------------------------------
    # block bootstrap
    # ----------------------------------------------------------------------------------------------------
    @staticmethod
    def block_resample(x, rng, block=20):
        """
        One circular moving-block resample of x, same length: runs of `block` consecutive days keep volatility
        clusters intact. rng is a np.random.Generator. Returns an ndarray of len(x).
        """
        n = len(x)
        nb = -(-n // block)
        return x[(rng.integers(0, n, nb)[:, None] + np.arange(block)[None, :]).ravel()[:n] % n]

    def bootstrap_shape(self, B=500, block=20, seed=0):
        """
        Block-bootstrap of the sample shape and skewness: B resampled histories of x, each built from random runs of
        `block` consecutive days (volatility clusters stay intact), with the statistics recomputed on each. No refit,
        so it is fast (~1 ms per resample). The spread across rows is the sampling uncertainty.

        Returns a pd.DataFrame of B rows with float columns v1, v2, skew. For example:
            np.cov(b[['v1', 'v2']].values.T)                  the plane's ellipse
            np.percentile(b.v2 - b.v1, [2.5, 97.5])           95% interval of v2 - v1
            np.percentile(b['skew'], [2.5, 97.5])             95% interval of the skew; one that excludes 0 is a real asymmetry
        Use b['skew'], not b.skew, which is the pandas DataFrame.skew method.
        """
        rng = np.random.default_rng(seed)
        rows = []
        for _ in range(B):
            xb = self.block_resample(self.x, rng, block)
            s = self.sample_shape(xb)
            rows.append((s['v1'], s['v2'], float(skew(xb))))
        return pd.DataFrame(rows, columns=['v1', 'v2', 'skew'])

    # ----------------------------------------------------------------------------------------------------
    # the fitted distribution: X = loc + c Z/V, loc = sample mean, c such that sd(X) = sample sd
    # ----------------------------------------------------------------------------------------------------
    @cached_property
    def _mixture(self):
        # X e^{E log V} = Z e^{-U}, U = log V - E log V, over the exact density of U (FracChiMean.log_v_pdf, the log
        # inversion), so it works even where GAS_SN cannot be built (the FCM scale underflows near the LNN)
        self._require_fit()
        f = FracChiMean(self.alpha, self.k)
        su = np.sqrt(f.log_var())
        u = np.linspace(-14*su, 14*su, 4001)
        fu = f.log_v_pdf(u)
        return u, fu, 1/np.sqrt(trapezoid(np.exp(-2*u)*fu, u))      # c/sd

    def pdf(self, xg):
        """The fitted GSaS density at the returns xg (an array). Returns an ndarray of len(xg)."""
        u, fu, c1 = self._mixture
        c = self.sd*c1
        y = (np.asarray(xg, dtype=float)[:, None] - self.mean)/c*np.exp(u)[None, :]
        return trapezoid(np.exp(u)[None, :]*norm.pdf(y)*fu[None, :], u, axis=1)/c

    def cdf(self, xg):
        """The fitted GSaS cdf at the returns xg (an array). Returns an ndarray of len(xg)."""
        u, fu, c1 = self._mixture
        c = self.sd*c1
        y = (np.asarray(xg, dtype=float)[:, None] - self.mean)/c*np.exp(u)[None, :]
        return trapezoid(norm.cdf(y)*fu[None, :], u, axis=1)

    def qq(self, n_grid=4001):
        """
        The Q-Q pairs at the plotting positions (i - 1/2)/n, the monotone cdf inverted on a grid of n_grid points.
        Returns a tuple (q_fit, x_sorted) of two ndarrays of len(x): the fitted and the sample quantiles.
        """
        xs = np.sort(self.x)
        p = (np.arange(1, len(xs) + 1) - 0.5)/len(xs)
        grid = np.linspace(xs[0] - 0.5*abs(xs[0]), xs[-1] + 0.5*abs(xs[-1]), n_grid)
        return np.interp(p, self.cdf(grid), grid), xs

    # ----------------------------------------------------------------------------------------------------
    # goodness of fit with the two standardized moments the fit does not use
    # ----------------------------------------------------------------------------------------------------
    @classmethod
    def v3_model(cls, alpha, k):
        """H(-3) as an implied log-variance, v3 = 2(K(-3) - 3/2 K(-2))/3. Needs E[V^-3]: k > 4 - alpha on k > 0."""
        K = cls.K_of(alpha, k)
        return 2*(K(-3) - 1.5*K(-2))/3

    @classmethod
    def vG_model(cls, alpha, k):
        """Geary's ratio H(-1) as an implied log-variance, vG = -2(K(-1) - 1/2 K(-2)). Needs E[V^-1]: k > 2 - alpha on k > 0."""
        K = cls.K_of(alpha, k)
        return -2*(K(-1) - 0.5*K(-2))

    @classmethod
    def v3_sample(cls, x):
        """The sample v3: 2/3 log of E|d|^3 / E[d^2]^1.5 relative to the normal's, d = x - mean(x)."""
        d = x - x.mean()
        return 2*np.log(np.mean(np.abs(d)**3)/np.mean(d**2)**1.5/cls.A3_NORMAL)/3

    @classmethod
    def vG_sample(cls, x):
        """The sample vG: -2 log of Geary's ratio E|d| / E[d^2]^0.5 relative to the normal's, d = x - mean(x)."""
        d = x - x.mean()
        return -2*np.log(np.mean(np.abs(d))/np.sqrt(np.mean(d**2))/cls.A1_NORMAL)

    @classmethod
    def gof_stats(cls, x, alpha=None, k=None):
        """
        The GoF statistics of x; (alpha, k) is refitted from x when not given. Returns a tuple of five floats
        (v3 sample, v3 GSaS, vG sample, vG GSaS, LNN reference (v1 + v2)/2), or None when the refit fails.
        """
        s = cls.sample_shape(x)
        if alpha is None:
            if s['exkurt'] <= 0:
                return None
            alpha, k = cls.invert(s['exkurt'], s['SPD'])
            if alpha is None:
                return None
        return cls.v3_sample(x), cls.v3_model(alpha, k), cls.vG_sample(x), cls.vG_model(alpha, k), (s['v1'] + s['v2'])/2

    def gof(self, B=200, block=20, seed=0):
        """
        Goodness of fit with H(-3) (third absolute moment) and H(-1) (Geary's ratio), each compared with the GSaS fit
        and with the LNN reference (v1 + v2)/2. z = gap / bootstrap sd, where the block bootstrap refits (alpha, k) on
        every resample (~0.2 s each); B = 0 skips it and the z columns are nan. |z| < 2 is within sampling noise.

        Returns a pd.DataFrame indexed by check ('H(-3): v3', 'H(-1): vG (Geary)') with float columns
        data, GSaS, LNN, z_GSaS, z_LNN. df.attrs['n_boot'] is the number of resamples that inverted.
        """
        self._require_fit()
        if 0 < self.k <= 4 - self.alpha:
            warnings.warn(f"E[V^-3] does not exist for k = {self.k:.3f} <= 4 - alpha: the H(-3) check is undefined")
        v3d, v3g, vGd, vGg, vl = self.gof_stats(self.x, self.alpha, self.k)
        rows = {'H(-3): v3': (v3d, v3g, vl), 'H(-1): vG (Geary)': (vGd, vGg, vl)}
        sds, n_boot = {name: (np.nan, np.nan) for name in rows}, 0
        if B:
            rng = np.random.default_rng(seed)
            b = np.array([r for r in (self.gof_stats(self.block_resample(self.x, rng, block)) for _ in range(B))
                          if r is not None])
            n_boot = len(b)
            sds = {'H(-3): v3': (np.std(b[:, 0] - b[:, 1], ddof=1), np.std(b[:, 0] - b[:, 4], ddof=1)),
                   'H(-1): vG (Geary)': (np.std(b[:, 2] - b[:, 3], ddof=1), np.std(b[:, 2] - b[:, 4], ddof=1))}
        df = pd.DataFrame([dict(check=name, data=vd, GSaS=vg, LNN=vl_, z_GSaS=(vd - vg)/sds[name][0], z_LNN=(vd - vl_)/sds[name][1])
                           for name, (vd, vg, vl_) in rows.items()]).set_index('check')
        df.attrs['n_boot'] = n_boot
        return df

    # ----------------------------------------------------------------------------------------------------
    # reference curves on the implied log-variance plane (for plotting; no matplotlib here)
    # ----------------------------------------------------------------------------------------------------
    @classmethod
    def plane_k_curve(cls, k, v_max):
        """
        The fixed-k curve on the plane, swept in alpha where the kurtosis exists (alpha > 5 - k, or alpha > 1 - |k|),
        up to v1 = 1.05 v_max. Returns a tuple (v1, v2) of two ndarrays, sorted by v1.
        """
        a = cls.ALPHAS[cls.ALPHAS > (5 - k if k > 0 else 1 - abs(k))]
        v1, v2 = cls.v12(a, k)
        ok = np.isfinite(v1) & np.isfinite(v2) & (v1 > 0) & (v1 <= v_max*1.05)
        order = np.argsort(v1[ok])
        return v1[ok][order], v2[ok][order]

    @classmethod
    def plane_student_t(cls, v_max, n=600):
        """
        Student's t (alpha = 1, nu = k > 4) on the plane in closed form, from v1 = v_max down to the origin.
        Returns a tuple (v1, v2) of two ndarrays, sorted by v1.
        """
        k_t = np.geomspace(4 + 6/(3*np.expm1(4*v_max)), 1000.0, n)
        g2_t, spd_t = 6/(k_t - 4), student_t.pdf(0, k_t)*np.sqrt(k_t/(k_t - 2))
        return (np.log1p(g2_t/3)/4)[::-1], (2*np.log(cls.SQ2PI*spd_t)/3)[::-1]
