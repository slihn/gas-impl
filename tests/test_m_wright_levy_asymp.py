
import numpy as np
import pandas as pd


from .wright import wright_m_fn_by_levy, wright_mainardi_fn_ts, wright_m_fn_find_x_by_asymp_gg, wright_m_fn_by_levy_find_x_by_step
from .wright_levy_asymp import wright_m_fn_by_levy_asymp, wright_f_fn_by_levy_asymp, get_x_by_large_coef_idx
from .unit_test_utils import *


# ----------------------------------------------------------------
class Test_Wright_M_Fn:
    small_alpha = [0.001, 0.01, 0.05, 0.09]
    medium_alpha_levy = [0.1, 0.25, 0.55, 0.75]
    large_alpha = [0.9, 0.93, 0.97, 0.99, 0.992, 0.993]  # beyond 0.994 is a different regime
    ultra_large_alpha = [0.994, 0.995, 0.996, 0.997, 0.998]
        
    def test_m_wright_levy(self):
        for alpha in self.small_alpha + self.medium_alpha_levy:
            x = np.array([0.0, 0.2, 0.5, 1.0, 2.0])
            p1 = wright_m_fn_by_levy_asymp(x, alpha)
            p2 = wright_m_fn_by_levy(x, alpha)
            for i in range(len(x)):
                delta_precise_up_to(float(p1[i]), float(p2[i]), msg_prefix=f"alpha={alpha}, x={x[i]}: ")  # type: ignore

    def test_small_fn_medium(self):
        # alpha = 0.1 has too much error, 5e-4
        medium_alpha_tweaked = [0.15, 0.2, 0.25, 0.55, 0.75]
        for alpha in medium_alpha_tweaked:
            for n in [4, 5, 6, 6.5]:
                x = wright_m_fn_find_x_by_asymp_gg(10**(-n), alpha)

                p1 = wright_m_fn_by_levy_asymp(x, alpha)
                p2 = wright_m_fn_by_levy(x, alpha)
                delta_precise_up_to(float(p1), float(p2), msg_prefix=f"alpha={alpha}, n={n}, x={x}: ")  # type: ignore

    def test_small_fn_small(self):
        for alpha in self.small_alpha:
            for n in [4, 5, 6, 6.5]:
                x = wright_m_fn_by_levy_find_x_by_step(10**(-n), alpha)

                p1 = wright_m_fn_by_levy_asymp(x, alpha)
                p2 = wright_m_fn_by_levy(x, alpha)
                delta_precise_up_to(float(p1), float(p2), reltol=1e-2, msg_prefix=f"alpha={alpha}, n={n}, x={x}: ")  # type: ignore

    def test_small_fn_large(self):
        for alpha in self.large_alpha:
            for n in [1, 2, 3]:
                x = get_x_by_large_coef_idx(alpha, n)

                p1 = wright_m_fn_by_levy_asymp(x, alpha)
                p2 = wright_m_fn_by_levy(x, alpha)
                assert isinstance(p2, float)
                assert p2 > 0.0, f"ERROR: alpha={alpha}, n={n}, x={x}: by_levy is zero"
                delta_precise_up_to(float(p1), float(p2), reltol=1e-2, msg_prefix=f"alpha={alpha}, n={n}, x={x}: ")  # type: ignore

    def test_small_fn_ultra_large(self):
        for alpha in self.ultra_large_alpha:
            for n in [1, 2]:  # this is only accurate to 1e-3, can not go beyond that
                x = get_x_by_large_coef_idx(alpha, n)

                p1 = wright_m_fn_by_levy_asymp(x, alpha)
                p2 = wright_mainardi_fn_ts(x, alpha)
                assert isinstance(p2, float)
                assert p2 > 0.0, f"ERROR: alpha={alpha}, n={n}, x={x}: by_ts is zero"
                delta_precise_up_to(float(p1), float(p2), reltol=1e-2, msg_prefix=f"alpha={alpha}, n={n}, x={x}: ")  # type: ignore

    def test_pdf_is_one(self):
        large_alpha = [0.9, 0.99, 0.993, 0.995, 0.998]
        for alpha in self.medium_alpha_levy + large_alpha:
            p1 = quad(lambda x: wright_m_fn_by_levy_asymp(x, alpha), 0, np.inf)[0]  # should be 1.0
            delta_precise_up_to(p1, 1.0, msg_prefix=f"alpha={alpha}: ")  # type: ignore


class Test_Wright_F_Fn:
    small_alpha = [0.001, 0.01, 0.05]
    medium_alpha_levy = [0.1, 0.25, 0.55, 0.75]

    def test_m_wright_levy(self):
        for alpha in self.small_alpha + self.medium_alpha_levy:
            for x in [0.2, 0.5, 1.0, 2.0]:
                p1 = wright_f_fn_by_levy_asymp(x, alpha)
                p2 = wright_m_fn_by_levy(x, alpha) * x * alpha  # type: ignore
                delta_precise_up_to(float(p1), float(p2), msg_prefix=f"alpha={alpha}, x={x}: ")  # type: ignore

    def test_small_fn(self):
        # alpha = 0.1 has too much error, 5e-4
        medium_alpha_tweaked = [0.15, 0.2, 0.25, 0.55, 0.75]
        for alpha in medium_alpha_tweaked:
            for n in [4, 5, 6, 6.5]:
                x = wright_m_fn_find_x_by_asymp_gg(10**(-n), alpha)
                p1 = wright_f_fn_by_levy_asymp(x, alpha)
                p2 = wright_m_fn_by_levy(x, alpha) * x * alpha  # type: ignore
                delta_precise_up_to(float(p1), float(p2), msg_prefix=f"alpha={alpha}, n={n}, x={x}: ")  # type: ignore
