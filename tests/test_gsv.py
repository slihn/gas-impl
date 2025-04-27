# test gsv

import numpy as np
import pandas as pd
from scipy.special import gamma
from scipy.integrate import quad

from .stable_count_dist import stable_count, gen_stable_count, stable_vol
from .unit_test_utils import *


# ----------------------------------------------------------------
def test_sv_known_case():
    alpha = 1.0
    sv = stable_vol(alpha)
    sc = stable_count(alpha/2)
    x = 0.85

    p = sv.pdf(x)  # type: ignore
    q1 = x * np.exp(-x**2/2)
    q2 = 2.0 * np.sqrt(2*np.pi) * sc.pdf(x**2 * 2)  # type: ignore
    delta_precise_up_to(p, q1)
    delta_precise_up_to(p, q2)


# ----------------------------------------------------------------
def test_sv_equal_gsc():
    alpha = 0.65
    x = 2.3
    sv = stable_vol(alpha)
    gsc = gen_stable_count(alpha=alpha/2, sigma=1.0/np.sqrt(2.0), d=1.0, p=alpha)
    compare_two_rvs(x, sv, gsc, min_p=0.1)


