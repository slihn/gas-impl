
# test gsas

import numpy as np
import pandas as pd
from scipy.special import gamma
from scipy.integrate import quad
from scipy.stats import tstd, levy_stable

from .frac_gamma_dist import frac_gamma
from .fcm_dist import frac_chi_mean, fcm_sigma, frac_chi2_mean
from .gas_dist import from_feller_to_s1, g_from_theta, levy_stable_rvs_from_ratio
from .tilted_stable import InverseStable, TitledStable, get_tilted_stable2, get_tilted_stable3, PitmanYorRestaurant
from .wright import M_Wright_One_Sided
from .unit_test_utils import *


# -------------------------------------------------------------------------------------
# very hard RVS here, will take a long time
debug = False

# -------------------------------------------------------------------------------------
class Test_M_Wright_RVS:
    alpha = 0.72
    mw = M_Wright_One_Sided(alpha)
    inv_stable = InverseStable(alpha)  # this is Kanter's method
    ts2_mw = get_tilted_stable2(alpha, 0)  # beta=0 gives S_alpha^{-alpha}
    ts3_mw = get_tilted_stable3(alpha, 0, alpha)  # beta=0, gamma=alpha gives S_alpha^{-alpha}
    NUM_STEPS = 5_000_000

    def test_mw_rvs(self):
        x1 = self.mw.rvs(self.NUM_STEPS)  # this is levy_stable_extremal
        delta_precise_up_to(np.mean(x1), self.mw.moment(1), abstol=0.001, reltol=0.001)
        delta_precise_up_to(tstd(x1), np.sqrt(self.mw.variance()), abstol=0.001, reltol=0.001)

    def test_inverse_stable_rvs(self):
        x2 = self.inv_stable.rvs(self.NUM_STEPS)
        delta_precise_up_to(np.mean(x2), self.mw.moment(1), abstol=0.001, reltol=0.001)
        delta_precise_up_to(tstd(x2), np.sqrt(self.mw.variance()), abstol=0.001, reltol=0.001)

    def test_ts2_rvs(self):
        x2 = self.ts2_mw.rvs(self.NUM_STEPS)
        delta_precise_up_to(np.mean(x2), self.mw.moment(1), abstol=0.001, reltol=0.001)
        delta_precise_up_to(tstd(x2), np.sqrt(self.mw.variance()), abstol=0.001, reltol=0.001)

    def test_ts3_rvs(self):
        x3 = self.ts3_mw.rvs(self.NUM_STEPS)
        delta_precise_up_to(np.mean(x3), self.mw.moment(1), abstol=0.001, reltol=0.001)
        delta_precise_up_to(tstd(x3), np.sqrt(self.mw.variance()), abstol=0.001, reltol=0.001)


# -------------------------------------------------------------------------------------
class Test_Frac_Gamma_RVS:
    alpha = 0.72
    sigma = 2.5
    d = 2.4
    p = 0.68

    beta = alpha * d / p
    gamma = alpha / p
    fg = frac_gamma(alpha, sigma, d, p)
    ts2 = get_tilted_stable2(alpha, beta)
    ts3 = get_tilted_stable3(alpha, beta, gamma)
    NUM_STEPS = 30_000_000  # it takes a lot more steps to converge !


    def test_mw_rvs(self):
        x1 = self.fg.rvs(self.NUM_STEPS)  # this is levy_stable_extremal
        delta_precise_up_to(np.mean(x1), self.fg.moment(1), abstol=0.001, reltol=0.001)
        delta_precise_up_to(tstd(x1), self.fg.std(), abstol=0.001, reltol=0.001)

    def test_ts2_rvs(self):
        x2 = self.ts2.fg_rvs(self.NUM_STEPS, self.sigma, self.p)
        delta_precise_up_to(np.mean(x2), self.fg.moment(1), abstol=0.001, reltol=0.001)
        delta_precise_up_to(tstd(x2), self.fg.std(), abstol=0.001, reltol=0.001)

    def test_ts3_rvs(self):
        x3 = self.ts3.rvs(self.NUM_STEPS) * self.sigma
        delta_precise_up_to(np.mean(x3), self.fg.moment(1), abstol=0.001, reltol=0.001)
        delta_precise_up_to(tstd(x3), self.fg.std(), abstol=0.001, reltol=0.001)


# -------------------------------------------------------------------------------------
class Test_FCM_Alpha1_RVS:
    alpha = 1.2
    k = 1.0
    scale = 1.5

    beta = 0.0
    gamma = 1/2
    fc = frac_chi_mean(alpha, k, scale=scale)  # this is fcm(alpha,1) for the stable distribution
    fc_sigma = 1/np.sqrt(2)
    ts3 = get_tilted_stable3(alpha/2, beta, gamma)
    NUM_STEPS = 30_000_000  # it takes a lot more steps to converge !

    def test_fcm_rvs(self):
        x1 = self.fc.rvs(self.NUM_STEPS)  
        delta_precise_up_to(np.mean(x1), self.fc.moment(1), abstol=0.001, reltol=0.001)
        delta_precise_up_to(tstd(x1), self.fc.std(), abstol=0.001, reltol=0.001)

    def test_ts3_rvs(self):
        print(f"Test_FCM_RVS: alpha {self.alpha:.3f} beta {self.beta:.3f} gamma {self.gamma:.3f}")
        print(f"Test_FCM_RVS: fcm_sigma = {self.fc_sigma:.3f}, scale = {self.scale:.3f}")
        x3 = self.ts3.rvs(self.NUM_STEPS) * self.fc_sigma * self.scale
        delta_precise_up_to(np.mean(x3), self.fc.moment(1), abstol=0.001, reltol=0.001)
        delta_precise_up_to(tstd(x3), self.fc.std(), abstol=0.001, reltol=0.001)


class Test_FCM_Alpha1_Theta_RVS:
    alpha = 1.2
    theta = 0.5
    g = g_from_theta(alpha, theta)

    k = 1.0
    scale = 1.5
    fc = frac_chi_mean(alpha, k, theta=theta, scale=scale)  # this is fcm(alpha,1) for the stable distribution


    beta = 0.0
    gamma = g
    fc_sigma = g**g
    ts3 = get_tilted_stable3(alpha*g, beta, gamma)
    NUM_STEPS = 30_000_000  # it takes a lot more steps to converge !

    def test_fcm_rvs(self):
        x1 = self.fc.rvs(self.NUM_STEPS)  
        delta_precise_up_to(np.mean(x1), self.fc.moment(1), abstol=0.001, reltol=0.001)
        delta_precise_up_to(tstd(x1), self.fc.std(), abstol=0.001, reltol=0.001)

    def test_ts3_rvs(self):
        print(f"Test_FCM_RVS: alpha {self.alpha:.3f} beta {self.beta:.3f} gamma {self.gamma:.3f}")
        print(f"Test_FCM_RVS: fcm_sigma = {self.fc_sigma:.3f}, scale = {self.scale:.3f}")
        x3 = self.ts3.rvs(self.NUM_STEPS) * self.fc_sigma * self.scale
        delta_precise_up_to(np.mean(x3), self.fc.moment(1), abstol=0.001, reltol=0.001)
        delta_precise_up_to(tstd(x3), self.fc.std(), abstol=0.001, reltol=0.001)


# -------------------------------------------------------------------------------------
class Test_FCM_RVS:
    alpha = 1.2
    k = 4.5
    scale = 1.5

    beta = (k-1)/2
    gamma = 1/2
    fc = frac_chi_mean(alpha, k, scale=scale)
    fc_sigma = fcm_sigma(alpha, k)
    ts3 = get_tilted_stable3(alpha/2, beta, gamma)
    NUM_STEPS = 30_000_000  # it takes a lot more steps to converge !

    def test_fcm_rvs(self):
        x1 = self.fc.rvs(self.NUM_STEPS)  
        delta_precise_up_to(np.mean(x1), self.fc.moment(1), abstol=0.001, reltol=0.001)
        delta_precise_up_to(tstd(x1), self.fc.std(), abstol=0.001, reltol=0.001)

    def test_ts3_rvs(self):
        if debug:
            print(f"Test_FCM_RVS: alpha {self.alpha:.3f} beta {self.beta:.3f} gamma {self.gamma:.3f}")
            print(f"Test_FCM_RVS: fcm_sigma = {self.fc_sigma:.3f}, scale = {self.scale:.3f}")
        x3 = self.ts3.rvs(self.NUM_STEPS) * self.fc_sigma * self.scale
        delta_precise_up_to(np.mean(x3), self.fc.moment(1), abstol=0.001, reltol=0.001)
        delta_precise_up_to(tstd(x3), self.fc.std(), abstol=0.001, reltol=0.001)


# -------------------------------------------------------------------------------------
class Test_FCM_NegK_RVS:
    alpha = 1.2
    k = 4.5
    scale = 1.5

    beta = k/2
    gamma = -1/2
    fc = frac_chi_mean(alpha, -k, scale=scale)
    fc_sigma = fcm_sigma(alpha, k)
    ts3 = get_tilted_stable3(alpha/2, beta, gamma)
    NUM_STEPS = 30_000_000  # it takes a lot more steps to converge !

    def test_fcm_rvs(self):
        x1 = self.fc.rvs(self.NUM_STEPS)  
        delta_precise_up_to(np.mean(x1), self.fc.moment(1), abstol=0.001, reltol=0.001)
        delta_precise_up_to(tstd(x1), self.fc.std(), abstol=0.001, reltol=0.001)

    def test_ts3_rvs(self):
        if debug:
            print(f"Test_FCM_NegK_RVS: fcm_sigma = {self.fc_sigma:.4f}, scale = {self.scale:.4f}")
        x3 = self.ts3.rvs(self.NUM_STEPS) / self.fc_sigma * self.scale
        delta_precise_up_to(np.mean(x3), self.fc.moment(1), abstol=0.001, reltol=0.001)
        delta_precise_up_to(tstd(x3), self.fc.std(), abstol=0.001, reltol=0.001)


# -------------------------------------------------------------------------------------
class Test_FCM2_RVS:
    alpha = 1.2
    k = 4.5
    scale = 1.5

    beta = (k-1)/2
    gamma = 1.0
    fc = frac_chi2_mean(alpha, k, scale=scale)
    fc_sigma = fcm_sigma(alpha, k)**2
    ts3 = get_tilted_stable3(alpha/2, beta, gamma)
    NUM_STEPS = 30_000_000  # it takes a lot more steps to converge !

    def test_fcm_rvs(self):
        x1 = self.fc.rvs(self.NUM_STEPS)  
        delta_precise_up_to(np.mean(x1), self.fc.moment(1), abstol=0.001, reltol=0.001)
        delta_precise_up_to(tstd(x1), self.fc.std(), abstol=0.001, reltol=0.001)

    def test_ts3_rvs(self):
        if debug:
            print(f"Test_FCM2_RVS: alpha {self.alpha:.3f} beta {self.beta:.3f} gamma {self.gamma:.3f}")
            print(f"Test_FCM2_RVS: fcm_sigma = {self.fc_sigma:.3f}, scale = {self.scale:.3f}")
        x3 = self.ts3.rvs(self.NUM_STEPS) * self.fc_sigma * self.scale
        delta_precise_up_to(np.mean(x3), self.fc.moment(1), abstol=0.001, reltol=0.001)
        delta_precise_up_to(tstd(x3), self.fc.std(), abstol=0.001, reltol=0.001)


# -------------------------------------------------------------------------------------
class Test_FCM2_NegK_RVS:
    alpha = 1.25
    k = 5.5
    scale = 1.5

    beta = k/2
    gamma = -1.0
    fc = frac_chi2_mean(alpha, -k, scale=scale)
    fc_sigma = fcm_sigma(alpha, k)**2
    ts3 = get_tilted_stable3(alpha/2, beta, gamma)
    NUM_STEPS = 30_000_000  # it takes a lot more steps to converge !

    fc_no_scale = frac_chi2_mean(alpha, -k, scale=fc_sigma)
    ts = TitledStable(alpha/2, beta)  # to validate its PDF

    def test_fcm_rvs(self):
        x1 = self.fc.rvs(self.NUM_STEPS)  
        delta_precise_up_to(np.mean(x1), self.fc.moment(1), abstol=0.01, reltol=0.01)
        delta_precise_up_to(tstd(x1), self.fc.std(), abstol=0.01, reltol=0.01)

    def test_ts3_rvs(self):
        if debug:
            print(f"Test_FCM2_NegK_RVS: fcm_sigma = {self.fc_sigma:.4f}, scale = {self.scale:.4f}")
        x3 = self.ts3.rvs(self.NUM_STEPS) / self.fc_sigma * self.scale
        delta_precise_up_to(np.mean(x3), self.fc.moment(1), abstol=0.01, reltol=0.01)
        delta_precise_up_to(tstd(x3), self.fc.std(), abstol=0.01, reltol=0.01)

    def test_ts_pdf(self):
        def fn0(x): return self.ts.pdf(x)  # type: ignore 
        m0 = quad(fn0, a=0.0001, b=np.inf, limit=10000)[0]
        delta_precise_up_to(m0, 1.0, msg_prefix="CDF=1 test")

        # test T_{alpha/2, k/2} is chi2_{alpha,-k}
        x_list = [1.0, 1.1]
        for x in x_list:  
            p1 = self.fc_no_scale.pdf(x)  # type: ignore
            delta_precise_up_to(p1, self.ts.pdf(x), msg_prefix=f"PDF test x={x}")


# -------------------------------------------------------------------------------------
class Test_Levy_Stable_Ratio_RVS:
    alpha = 1.2
    theta = 0.5

    beta, scale = from_feller_to_s1(alpha, theta)
    levy = levy_stable(alpha, beta=beta, scale=scale)

    x = 0.5
    hist_points = np.array([-x, 0.0, x], dtype=float)
    NUM_STEPS = 30_000_000  # it takes a lot more steps to converge !
    HIST_QUANTILES = (0.005, 0.995)
    BINS = 1000

    def centers_and_density_from_rvs(self, samples):
        finite_samples = samples[np.isfinite(samples)]
        x_min, x_max = np.quantile(finite_samples, self.HIST_QUANTILES)  # prevent tails to mess up histogram
        edges = np.linspace(x_min, x_max, self.BINS + 1)
        widths = np.diff(edges)
        counts, _ = np.histogram(finite_samples, bins=edges)
        density = counts / (finite_samples.size * widths)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers, density

    def compare_pdf_hist(self, samples, msg, abstol=0.01, reltol=0.01):
        centers, density = self.centers_and_density_from_rvs(samples)
        for x in self.hist_points:
            pdf = self.levy.pdf(x)  # type: ignore
            hist_density = np.interp(x, centers, density, left=np.nan, right=np.nan)
            delta_precise_up_to(hist_density, pdf, abstol=abstol, reltol=reltol, msg_prefix=f"{msg} x={x}:")

    def test_scipy_rvs(self):
        x1 = self.levy.rvs(self.NUM_STEPS)  
        self.compare_pdf_hist(x1, "levy_stable scipy rvs:")

    def test_ratio_rvs(self):
        x2 = levy_stable_rvs_from_ratio(self.NUM_STEPS, self.alpha, self.theta) 
        self.compare_pdf_hist(x2, "levy_stable ratio rvs:")


# -------------------------------------------------------------------------------------
class Test_Pitman_Yor:
    alpha = 1.2
    k = 4.5
    NUM_CUSTOMERS_LARGE = 20_000_000
    NUM_CUSTOMERS_SMALL = 100_000
    NUM_PATHS = 50_000

    fc = frac_chi_mean(alpha, k)
    fc_scale = fcm_sigma(alpha, k)

    pyr = None # this should be populated later
    ks = None  # this should be populated later

    def _get_pyr(self, num_customers):
        return PitmanYorRestaurant(
            alpha=self.alpha/2,
            beta=(self.k-1)/2,
            gamma=0.5,  # FCM
            num_customers=num_customers)        

    def _generate_py_samples(self):
        if self.pyr is not None and self.ks is not None:
            return

        # this is NUM_CUSTOMERS_SMALL x NUM_PATHS
        self.pyr = self._get_pyr(self.NUM_CUSTOMERS_SMALL)
        self.ks, self.py_samples, self.kanter_samples = self.pyr.ks_kanter_crp(
            size=self.NUM_PATHS,
            return_samples=True,
        )

    def test_py_samples(self):
        # KS test, statistic less than 1%
        self._generate_py_samples()
        assert isinstance(self.pyr, PitmanYorRestaurant)

        exact_mean = self.pyr.mean()
        exact_std = self.pyr.std()

        p1 = self.ks.statistic * 100.0  # type: ignore
        p2 = 1.0
        assert p1 < 1.0, f"ERROR: ks statistics {p1} pct is larger than {p2}"

        py_mean = self.py_samples.mean()  # type: ignore
        py_std = self.py_samples.std(ddof=1)  # type: ignore

        kanter_mean = self.kanter_samples.mean()  # type: ignore
        kanter_std = self.kanter_samples.std(ddof=1)  # type: ignore

        delta_precise_up_to(py_mean, exact_mean, abstol=0.005, reltol=0.005)
        delta_precise_up_to(kanter_mean, exact_mean, abstol=0.005, reltol=0.005)

        delta_precise_up_to(py_std, exact_std, abstol=0.05, reltol=0.05)
        delta_precise_up_to(kanter_std, exact_std, abstol=0.05, reltol=0.05)

    def test_fcm_moments(self):
        self._generate_py_samples()

        fcm_py_samples = self.fc_scale * self.py_samples
        py_mean2 = fcm_py_samples.mean()
        py_std2 = fcm_py_samples.std(ddof=1)

        delta_precise_up_to(py_mean2, self.fc.moment(1), abstol=0.005, reltol=0.005)
        delta_precise_up_to(py_std2, self.fc.std(), abstol=0.05, reltol=0.05)
     
    def test_path_ensemble(self):
        # millions of customers, small amount of paths
        # this is NUM_CUSTOMERS_LARGE x num_paths
        num_paths = 200
        num_checkpoints = 100
        pyr2 = self._get_pyr(self.NUM_CUSTOMERS_LARGE)
        expected_terminal_val = pyr2.mean()

        path_df = pyr2.pitman_yor_paths(size=num_paths, num_checkpoints=num_checkpoints)
        max_customers = path_df.customers.max()
        path_terminal_avg = path_df.query("customers == @max_customers")['tilted-stable estimate'].mean()

        delta_precise_up_to(path_terminal_avg, expected_terminal_val, abstol=0.05, reltol=0.05)

    def test_stick_breaking_rvs(self):
        """Fails when the GEM beta parameters or residual scaling are incorrect."""
        pyr = self._get_pyr(1)
        ks, stick_samples, kanter_samples = pyr.ks_kanter_stick(
            size=40_000,
            num_breaks=2_000,
            return_samples=True,
        )
        stick_mean = stick_samples.mean()  # type: ignore
        stick_std = stick_samples.std(ddof=1)  # type: ignore

        assert np.all(np.isfinite(stick_samples))
        assert np.all(np.isfinite(kanter_samples))
        assert np.isclose(stick_mean, pyr.mean(), rtol=0.005)
        assert np.isclose(stick_std, pyr.std(), rtol=0.02)
        assert ks.statistic < 0.01  # type: ignore
