
# test gsc and sc

import numpy as np
import pandas as pd

from scipy.special import gamma, hyp1f1, gammainc
from scipy.integrate import quad

from .frac_gamma import frac_gamma_star_by_hyp1f1, frac_gamma_star_by_m, frac_gamma_star_by_f, frac_gamma_inc, frac_gamma_inc_by_m
from .unit_test_utils import *


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