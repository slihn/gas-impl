
# test gsc and sc

import numpy as np
import pandas as pd

from scipy.special import gamma, hyp1f1, gammainc
from scipy.integrate import quad

from .frac_gamma import frac_gamma_star_by_hyp1f1, frac_gamma_star_by_m, frac_gamma_star_supplementary_by_m,\
    frac_gamma_star_by_f, frac_gamma_inc, frac_gamma_inc_by_m, frac_gamma_star_total
from .unit_test_utils import *

from pandarallel import pandarallel

pandarallel.initialize(verbose=1)


class Test_Gamma_Star:
    alpha = 0.46 
    x = 0.85
    s = 2.4

    p1 = frac_gamma_star_by_m(s, x, alpha)
    
    def test_gamma_star_by_f(self):
        p2 = frac_gamma_star_by_f(self.s, self.x, self.alpha)
        delta_precise_up_to(self.p1, p2)

    def test_gamma_star_by_hyp1f1(self):
        p2 = frac_gamma_star_by_hyp1f1(self.s, self.x, self.alpha)
        delta_precise_up_to(self.p1, p2)

    def test_gamma_star_by_hyp1f1_int(self):
        p2 = frac_gamma_star_by_hyp1f1(self.s, self.x, self.alpha, use_int=True, by_levy=False)
        delta_precise_up_to(self.p1, p2)
        p3 = frac_gamma_star_by_hyp1f1(self.s, self.x, self.alpha, use_int=True, by_levy=True)
        delta_precise_up_to(self.p1, p3)

    def test_gamma_star0_by_frac_hyp1f1_m(self):
        p1 = gammainc(self.s, self.x) * self.x**(-self.s)
        p2 = frac_gamma_star_by_hyp1f1(self.s, self.x, 0.0)
        delta_precise_up_to(p1, p2)

    def test_gamma_star0_by_m(self):
        p1 = gammainc(self.s, self.x) * self.x**(-self.s)
        p2 = frac_gamma_star_by_m(self.s, self.x, 0.0)
        delta_precise_up_to(p1, p2)

    # test supplementary function
    def test_gamma_star_supplementary_total(self):
        p1 = frac_gamma_star_supplementary_by_m(self.s, self.x, self.alpha, a=0.0, b=np.inf)
        p2 = self.x**(-self.s)
        delta_precise_up_to(p1, p2)

    def test_gamma_star_supplementary(self):
        p2 = self.x**(-self.s) - self.p1
        p3 = frac_gamma_star_supplementary_by_m(self.s, self.x, self.alpha)
        delta_precise_up_to(p2, p3)


def test_frac_gamma_star_total():
    # total = 1 at x = 1 is important since we are using this in cdf of frac gamma distribution
    x = 1.0
    df = pd.DataFrame(
        [ (s, alpha)
            for s in np.linspace(0.1, 10.0, 40)
            for alpha in np.linspace(0.01, 0.99, 40)
            if not (s > 5 and alpha <= 0.1)  # large s doesn't work well with small alpha
        ],
        columns=['s', 'alpha']
    )

    df['total_fg'] = df.parallel_apply(lambda row: frac_gamma_star_total(row.s, x, row.alpha), axis=1)  # type: ignore
    df['error'] = np.abs(df['total_fg'] - 1.0)  # type: ignore
    bad_df = df[df['error'] > 1e-4]
    assert len(bad_df) == 0, f"ERROR: frac_gamma_star_total failed for s/alpha:\n{bad_df}"


class Test_Gamma_Inc:
    alpha = 0.6 
    x = 0.85
    s = 1.5

    p1 = frac_gamma_inc(s, x, alpha)
    
    def test_gamma_inc_by_m(self):
        p2 = frac_gamma_inc_by_m(self.s, self.x, self.alpha)
        delta_precise_up_to(self.p1, p2)
        
    def test_gamma_inc_0(self):
        p2 = frac_gamma_inc_by_m(self.s, 0, self.alpha)
        delta_precise_up_to(0.0, p2)
        
    def test_gamma_inc_1(self):
        # s needs to be small enough
        p2 = frac_gamma_inc(self.s, 10.0, self.alpha)
        delta_precise_up_to(1.0, p2)
        

def test_frac_gamma_inc_half():
    alpha = 0.5 
    x = 0.85
    s = 1.5
    p1 = frac_gamma_inc(s, x, alpha)
    p2 = gammainc(s, x)
    delta_precise_up_to(p1, p2)