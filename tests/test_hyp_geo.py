
# test hyp geo

import numpy as np
import pandas as pd
from typing import Union, List, Optional

from scipy.special import gamma, hyp1f1, hyp2f1, poch
from scipy.integrate import quad
from scipy.stats import norm

from .wright import mainardi_wright_fn
from .hyp_geo import *
from .hyp_geo2 import *
from .mellin import pdf_by_mellin
from .unit_test_utils import *


# ----------------------------------------------------------------
# basic stuff

def test_poch():
    n = 5
    s = 1.6
    p = 1.0/(n+s)
    q = poch(s, n) / poch(s+1, n) / s
    delta_precise_up_to(p, q)


def test_kummer_mellin():
    a = 2.3
    b = 3.4
    x = 0.35
    p1 = hyp1f1(a, b, -x) 
    p2 = pdf_by_mellin(x, lambda s: hyp1f1_mellin_transform(s, a=a, b=b))
    delta_precise_up_to(p1, p2)

def test_gauss_mellin():
    a = 2.3
    b = 3.4
    c = 1.5
    x = 0.35
    p1 = hyp2f1(a, b, c, -x) 
    p2 = pdf_by_mellin(x, lambda s: hyp2f1_mellin_transform(s, a=a, b=b, c=c))
    delta_precise_up_to(p1, p2)

# ----------------------------------------------------------------

def test_frac_hyp1f1_series_vs_int():
    alpha = 0.45
    x = [ -0.85, -0.75 ]
    a = 2.3
    b = 3.4
    p: List = frac_hyp1f1_m_int(x, alpha, a, b) 
    q: List = frac_hyp1f1_m(x, alpha, a, b)  # type: ignore
    delta_precise_up_to(p[0], q[0])
    delta_precise_up_to(p[1], q[1])


class Test_Frac_Hyp1f1_to_M_Wright:
    x = 0.85
    alpha = 0.45

    def test_m_wright_vs_frac_hyp1f1_m(self):
        p1 = mainardi_wright_fn(self.x, self.alpha) 
        p2 = frac_hyp1f1_m(-self.x, self.alpha, 1.0, 1.0) 
        delta_precise_up_to(p1, p2)

    def test_m_wright_vs_Frac_Hyp1f1_M(self):
        p1 = mainardi_wright_fn(self.x, self.alpha) 
        p2 = Frac_Hyp1f1_M(self.alpha, 1.0, 1.0).series(-self.x)
        delta_precise_up_to(p1, p2)


    def test_frac_hyp1f1_vs_hyp1f1(self):
        x = -0.85
        a = 2.3
        b = 3.4
        p1 = hyp1f1(a, b, x) 
        p2 = frac_hyp1f1_m(x, 0.0, a, b) 
        delta_precise_up_to(p1, p2)



class Test_Frac_Hyp1f1_to_others:
    def test_frac_hyp1f1_series_vs_int(self):
        a = 0.5
        b = 1.5 
        x = -0.35
        p1 = hyp1f1(a, b, x)
        p2 = Frac_Hyp1f1(0.0, 1.0, a=a, b=b).integral(x)
        p3 = Frac_Hyp1f1(0.0, 1.0, a=a, b=b).series(x)
        delta_precise_up_to(p1, p2)
        delta_precise_up_to(p1, p3)


# ----------------------------------------------------------------
class Test_Frac_Hyp2f1_Alpha_1:
    a = 1.5
    b = 3.4
    c = 1.6
    
    eps = 1.0
    fh = Frac_Hyp2f1(a, b, c, eps=eps)
    
    def test_fcm2_hat(self):
        s = 0.35
        p1 = self.fh.fcm2_hat_mellin_transform(s)  # type: ignore
        k2 = self.fh.k / 2.0
        p2 = gamma(s + k2 -1) / gamma(k2)
        delta_precise_up_to(p1, p2)  # type: ignore

    def test_gauss_mellin_equiv_a1(self):
        s = 0.25
        p1 = self.fh.mellin_transform(s) 
        p2 = hyp2f1_mellin_transform(s, a=self.fh.a, b=self.fh.b, c=self.fh.c)
        delta_precise_up_to(p1, p2)

    def test_integral(self):
        x = -0.45
        p1 = self.fh.integral(x)
        p2 = hyp2f1(self.a, self.b, self.c, x)
        delta_precise_up_to(p1, p2)
        

class Test_Frac_Hyp2f1:
    a = 0.6
    b = 3.2
    c = 1.5
    
    eps = 0.9
    fh = Frac_Hyp2f1(a, b, c, eps=eps)
 
    def test_integral(self):
        x = -0.45
        p1 = self.fh.integral(x)
        p2 = self.fh.integral_by_mellin(x)
        delta_precise_up_to(p1, p2)
        
    def test_gauss_mellin_equiv(self):
        s = 0.25
        p1 = self.fh.mellin_transform(s) 
        p2 = self.fh.mellin_transform_expanded(s)
        delta_precise_up_to(p1, p2)

    def test_scaled_integral_fcm2(self):
        x = -0.45
        p1 = self.fh.scaled_integral(x)
        p2 = self.fh.scaled_integral_by_fcm2(x)
        delta_precise_up_to(p1, p2)

    def test_scaled_integral_fcm(self):
        x = -0.45
        p1 = self.fh.scaled_integral(x)
        p3 = self.fh.scaled_integral_by_fcm(x)
        delta_precise_up_to(p1, p3)
