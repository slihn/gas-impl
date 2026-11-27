# Tests for GSaS_JumpRegimes (gsas_regime.py): the family routing per regime, GSaS -> LNEP -> normal.
#
# The data are simulated in two long blocks the jump model separates by volatility, followed by a turbulent
# Student-t(5) block, which GSaS fits. In the first test the calm block is LNEP(3.5, 0.25), whose flat peak
# (v2 = -0.065) no GSaS can match but LNEP can. In the second it is an EP(6) body with 2% of days at 3x the
# volatility scattered at random (too sparse for the jump model to split off): a flat top with heavy tails, gap
# v1 - v2 = 0.27, far beyond the LNEP band (0.088), which should be drawn as a normal with the hidden-regime note.

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from .gsas_regime import GSaS_JumpRegimes
from .lnep_dist import lnep, LNEP_Fit


def _two_blocks(calm, rng):
    wild = 0.02*stats.t.rvs(5, size=4000, random_state=rng)/np.sqrt(5/3)
    xs = pd.Series(np.concatenate([calm, wild]), index=pd.bdate_range('1990-01-01', periods=len(calm) + 4000))
    reg = GSaS_JumpRegimes(xs, n_regimes=2, jump_penalty=200).fit()
    n = len(calm)
    assert (reg.labels.values[:n] == 1).mean() > 0.95 and (reg.labels.values[n:] == 2).mean() > 0.95
    return reg, reg.summary()


def test_a_flat_peaked_regime_is_drawn_as_lnep_by_maximum_likelihood():
    rng = np.random.default_rng(4)
    reg, s = _two_blocks(0.004*lnep.rvs(3.5, 0.25, size=8000, random_state=rng), rng)
    assert s.loc[1, 'status'] != 'ok' and s.loc[1, 'drawn_as'] == 'LNEP' and s.loc[1, 'lnep_status'] == 'ok'
    assert s.loc[1, 'p_lnep'] >= 0.05 and s.loc[1, 'note'] == ''
    assert s.loc[1, 'lnep_alpha'] == pytest.approx(3.5, abs=1.0) and s.loc[1, 's'] == pytest.approx(0.25, abs=0.08)
    m = LNEP_Fit(reg.xs.values[reg.labels.values == 1])                       # the regime's fit is the MLE
    assert (s.loc[1, 'lnep_alpha'], s.loc[1, 's']) == (pytest.approx(m.alpha), pytest.approx(m.s))
    assert s.loc[2, 'drawn_as'] == 'GSaS' and s.loc[2, 'lnep_status'] == '' and np.isnan(s.loc[2, 'p_lnep'])
    g = np.linspace(-0.25, 0.25, 20001)
    assert np.trapz(reg.pdf(g), g) == pytest.approx(1.0, abs=2e-3)
    np.testing.assert_allclose(reg.component_pdfs(g)[:, 0], reg.weights[0]*reg.lnep_fits[0].pdf(g))


def test_a_flat_top_with_heavy_tails_is_drawn_as_normal_with_the_hidden_regime_note():
    rng = np.random.default_rng(4)
    body = lnep.rvs(6.0, 0.0, size=6000, random_state=rng)
    spike = rng.random(6000) < 0.02
    body[spike] = 3.0*rng.standard_normal(spike.sum())
    reg, s = _two_blocks(0.004*body, rng)
    assert s.loc[1, 'v2'] < 0 and s.loc[1, 'v1'] - s.loc[1, 'v2'] > 0.2
    assert s.loc[1, 'drawn_as'] == 'normal' and s.loc[1, 'p_lnep'] < 0.05 and s.loc[1, 'lnep_status'] == 'p < 0.05'
    assert 'hides a sharp regime' in s.loc[1, 'note'] and reg.lnep_fits[0] is None


def test_simulated_days_follow_the_regime_chain_and_each_regime_law():
    """simulate() is a hidden Markov model of the fit: the regime path switches at the rates of transition_matrix(),
    and each regime's days follow the law the mixture uses for it (LNEP through LNEP_Fit.rvs_of, GSaS through
    GAS_SN at beta = 0, checked against GSaS_Fit's own cdf)."""
    rng = np.random.default_rng(5)
    calm = 0.004*lnep.rvs(3.5, 0.25, size=(4, 1500), random_state=rng)
    wild = 0.02*stats.t.rvs(5, size=(4, 1500), random_state=rng)/np.sqrt(5/3)
    x = np.concatenate([b for pair in zip(calm, wild) for b in pair])           # 8 alternating blocks, 7 switches
    reg = GSaS_JumpRegimes(pd.Series(x, index=pd.bdate_range('1990-01-01', periods=len(x))),
                           n_regimes=2, jump_penalty=200).fit()
    assert reg.summary().drawn_as.tolist() == ['LNEP', 'GSaS']
    sim = reg.simulate(n=300_000, seed=1)
    assert sim.equals(reg.simulate(n=300_000, seed=1))
    lab = sim.regime.values
    P_sim = pd.crosstab(lab[:-1], lab[1:], normalize='index').values
    np.testing.assert_allclose(P_sim, reg.transition_matrix().values, rtol=0.3)   # ~200 switches each way
    for j, law in [(1, reg.lnep_fits[0]), (2, reg.fits[1])]:
        assert stats.kstest(sim.x.values[lab == j][:3000], law.cdf).pvalue > 0.01
