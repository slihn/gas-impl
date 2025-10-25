
from doctest import debug
import numpy as np
import pandas as pd
import mpmath as mp

from .wright import wright_m_fn_by_levy, wright_f_fn_by_levy, wright_m_fn_elasticity_by_levy, mainardi_wright_fn_slope
from .wright_mp import wright_m_fn_mp, wright_f_fn_mp, wright_m_fn_mp_elasticity, wright_m_fn_mp_slope
from .unit_test_utils import *


# ----------------------------------------------------------------
class Test_Wright_M_Fn:
    def test_m_wright_levy(self):
        mp.mp.prec = 128
        for alpha in [0.1, 0.25, 0.55, 0.75]:
            x = np.array([0.0, 0.2, 0.5, 1.0, 2.0])
            p1 = wright_m_fn_mp(x, alpha)
            p2 = wright_m_fn_by_levy(x, alpha)
            for i in range(len(x)):
                delta_precise_up_to(float(p1[i]), float(p2[i]), msg_prefix=f"alpha={alpha}, x={x[i]}: ")  # type: ignore

    def test_m_wright_elasticity(self):
        mp.mp.prec = 128
        for alpha in [0.1, 0.25, 0.55, 0.75]:
            x = np.array([0.0, 0.2, 0.5, 0.7, 1.0])
            p1 = wright_m_fn_mp_elasticity(x, alpha, d_log_x=mp.mpf(0.00001))
            p2 = wright_m_fn_elasticity_by_levy(x, alpha)
            for i in range(len(x)):
                delta_precise_up_to(float(p1[i]), float(p2[i]), msg_prefix=f"alpha={alpha}, x={x[i]}: ")  # type: ignore

    def test_m_wright_slope(self):
        mp.mp.prec = 128
        for alpha in [0.1, 0.25, 0.55, 0.75]:
            x = np.array([0.2, 0.5, 0.7, 1.0])
            p1 = wright_m_fn_mp_slope(x, alpha)
            p2 = mainardi_wright_fn_slope(x, alpha)
            for i in range(len(x)):
                delta_precise_up_to(float(p1[i]), float(p2[i]), msg_prefix=f"alpha={alpha}, x={x[i]}: ")  # type: ignore


class Test_Wright_F_Fn:
    def test_m_wright_levy(self):
        mp.mp.prec = 128
        for alpha in [0.1, 0.25, 0.55, 0.75]:
            for x in [0.2, 0.5, 1.0, 2.0]:
                p1 = wright_f_fn_mp(x, alpha)
                p2 = wright_f_fn_by_levy(x, alpha)
                delta_precise_up_to(float(p1), float(p2), msg_prefix=f"alpha={alpha}, x={x}: ")  # type: ignore

