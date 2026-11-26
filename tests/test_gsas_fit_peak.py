# Tests for GSaS_Fit.peak_report / peak_density: the centrality test and the widened peak on a central dip.
#
# References are closed-form densities of simulated laws: the normal peak 1/sqrt(2 pi), and for a symmetric
# two-normal mixture with a dip at 0, the mean density over a central window, which the widened peak must reach.

import numpy as np
import pytest
from scipy.stats import norm

from .gsas_fit import GSaS_Fit


def test_single_peak_keeps_the_extrapolation():
    x = np.random.default_rng(1).standard_normal(20000)
    r = GSaS_Fit.peak_report(x)
    assert not r['dip'] and r['h_peak'] == 0.0
    assert r['peak'] == r['extrap'] == GSaS_Fit.peak_density(x, GSaS_Fit.PEAK_HS)
    assert r['peak'] == pytest.approx(norm.pdf(0), rel=0.03)


def test_central_dip_widens_to_the_sparsity_maximum():
    # 0.5 N(-0.6, 0.5^2) + 0.5 N(0.6, 0.5^2): density 0.38 at 0, rising to 0.47 near +-0.55
    rng = np.random.default_rng(2)
    n = 20000
    x = rng.choice([-0.6, 0.6], n) + 0.5*rng.standard_normal(n)
    r = GSaS_Fit.peak_report(x)
    assert r['dip'] and r['peak'] > r['extrap'] and r['h_peak'] > 0.1
    # the widened peak is the best window average: exact over the central 2h for the true law, maximized over h
    f = lambda w: 0.5*(norm.cdf(w, 0.6, 0.5) - norm.cdf(-w, 0.6, 0.5) + norm.cdf(w, -0.6, 0.5) - norm.cdf(-w, -0.6, 0.5))/(2*w)
    true_best = max(f(w) for w in np.linspace(0.01, 2, 2000) if 0.5*(f(w)*2*w) <= GSaS_Fit.PEAK_WIDEN_HS[-1])
    assert r['peak'] == pytest.approx(true_best, rel=0.02)
    assert GSaS_Fit.peak_density(x) == r['peak']
