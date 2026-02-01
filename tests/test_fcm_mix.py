
# test fcm extensions from gaussian mixture chapter

import numpy as np
import pandas as pd

from .fcm_dist import fcm_moment, frac_chi_mean, fcm_sigma,\
    fcm_characteristic, fcm_characteristic_pdf, fcm_characteristic_pdf_by_extremal,\
    fcm_inverse, fcm_inverse_pdf, fcm_inverse_pdf_by_extremal,\
    fcm_characteristic_inverse, fcm_characteristic_inverse_pdf
from .unit_test_utils import *



# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
class Test_Characteristic_FCM:
    alpha = 1.25
    k = 2.8  # this is sensitive to k, k should be larger than 2.5

    f1 = frac_chi_mean(alpha, -k)
    f2 = fcm_characteristic(alpha, k)
    m1 = 1.0 / fcm_moment(1.0, alpha, k)

    def test_m1(self):
        delta_precise_up_to(self.f1.moment(1), self.m1)
        delta_precise_up_to(self.f2.moment(1), self.m1)

    def test_pdf(self):
        x = 1.5
        p1 = self.f1.pdf(x)  # type: ignore
        p2 = self.f2.pdf(x)  # type: ignore
        delta_precise_up_to(p1, p2, msg_prefix="f1 vs f2")

        p3 = fcm_characteristic_pdf(x, self.alpha, self.k)
        delta_precise_up_to(p1, p3, msg_prefix="f1 vs fcm_characteristic_pdf via relation")

        p4 = fcm_characteristic_pdf(x, self.alpha, self.k, use_relation=False)
        delta_precise_up_to(p1, p4, msg_prefix="f1 vs fcm_characteristic_pdf direct")

    def test_pdf_by_extremal(self):
        x = 1.2
        p1 = self.f1.pdf(x)  # type: ignore
        p4 = fcm_characteristic_pdf_by_extremal(x, self.alpha, self.k)
        delta_precise_up_to(p1, p4)

    def test_cdf(self):
        x = self.f2.ppf(0.2)
        p1 = self.f1.cdf(x)  # type: ignore
        p2 = self.f2.cdf(x)  # type: ignore
        delta_precise_up_to(p1, p2, msg_prefix="f1 vs f2")
        
        p3 = quad(lambda x: self.f2.pdf(x), a=0, b=x, limit=10000)[0]  # type: ignore
        delta_precise_up_to(p1, p3, msg_prefix="f2 vs quad")

    def test_cdf_near_zero(self):
        x = 0.01  # sensitive to x, can not be too small
        cdf1 = self.f1.cdf(x)  # type: ignore
        cdf2 = self.f2.cdf(x)  # type: ignore
        assert 0.0 <= cdf1 < 0.01, f"ERROR: cdf1 near zero not small enough: {cdf1} at x = {x}"
        assert 0.0 <= cdf2 < 0.01, f"ERROR: cdf2 near zero not small enough: {cdf2} at x = {x}"

    def test_cdf_at_large_x(self):
        x = 20.0
        cdf1 = self.f1.cdf(x)  # type: ignore
        cdf2 = self.f2.cdf(x)  # type: ignore
        assert 0.0 <= 1 - cdf1 < 0.01, f"ERROR: cdf1 at large x not close to 1: {cdf1}"
        assert 0.0 <= 1 - cdf2 < 0.01, f"ERROR: cdf2 at large x not close to 1: {cdf2}"


class Test_Inverse_FCM:
    alpha = 1.2
    k = 3.1

    f1 = fcm_inverse(alpha, k)

    def test_pdf_by_relation(self):
        x = 1.8
        p1 = self.f1.pdf(x)  # type: ignore
        p2 = fcm_inverse_pdf(x, self.alpha, self.k)
        delta_precise_up_to(p1, p2)

    def test_pdf_direct(self):
        x = 1.4
        p1 = self.f1.pdf(x)  # type: ignore
        p3 = fcm_inverse_pdf(x, self.alpha, self.k, use_relation=False)
        delta_precise_up_to(p1, p3)

    def test_pdf_by_extremal(self):
        x = 1.2
        p1 = self.f1.pdf(x)  # type: ignore
        p4 = fcm_inverse_pdf_by_extremal(x, self.alpha, self.k)
        delta_precise_up_to(p1, p4)

    def test_inverse_is_rescaled_characteristic(self):
        x = 1.25
        scale = fcm_sigma(self.alpha, self.k - 1) / fcm_sigma(self.alpha, self.k)
        p1 = self.f1.pdf(x)  # type: ignore
        p2 = fcm_characteristic(self.alpha, self.k - 1, scale=scale).pdf(x)  # type: ignore
        delta_precise_up_to(p1, p2, msg_prefix="inverse vs characteristic pdf with k+1")

    def test_cdf(self):
        x = self.f1.ppf(0.2)
        p1 = self.f1.cdf(x)  # type: ignore
        p2 = quad(lambda x: self.f1.pdf(x), a=0, b=x, limit=10000)[0]  # type: ignore
        delta_precise_up_to(p1, p2, msg_prefix="f1 vs quad")

    def test_cdf_near_zero(self):
        x = 0.01
        cdf1 = self.f1.cdf(x)  # type: ignore
        assert 0.0 <= cdf1 < 0.01, f"ERROR: cdf1 near zero not small enough: {cdf1}"

    def test_cdf_at_large_x(self):
        x = 20.0
        cdf1 = self.f1.cdf(x)  # type: ignore
        assert 0.0 <= 1 - cdf1 < 0.01, f"ERROR: cdf1 at large x not close to 1: {cdf1}"


class Test_Characteristic_Inverse_FCM:
    alpha = 1.2
    k = 3.1

    f1 = fcm_characteristic_inverse(alpha, k)

    def test_pdf_by_relation(self):
        x = 0.9
        p1 = self.f1.pdf(x)  # type: ignore
        p2 = fcm_characteristic_inverse_pdf(x, self.alpha, self.k)
        delta_precise_up_to(p1, p2)

    def test_pdf_direct(self):
        x = 1.1
        p1 = self.f1.pdf(x)  # type: ignore
        p3 = fcm_characteristic_inverse_pdf(x, self.alpha, self.k, use_relation=False)
        delta_precise_up_to(p1, p3)

    def test_cdf(self):
        x = self.f1.ppf(0.2)
        p1 = self.f1.cdf(x)  # type: ignore
        p2 = quad(lambda x: self.f1.pdf(x), a=0, b=x, limit=10000)[0]  # type: ignore
        delta_precise_up_to(p1, p2, msg_prefix="f1 vs quad")

    def test_cdf_near_zero(self):
        x = 0.01
        cdf1 = self.f1.cdf(x)  # type: ignore
        assert 0.0 <= cdf1 < 0.01, f"ERROR: cdf1 near zero not small enough: {cdf1}"

    def test_cdf_at_large_x(self):
        x = 20.0
        cdf1 = self.f1.cdf(x)  # type: ignore
        assert 0.0 <= 1 - cdf1 < 0.01, f"ERROR: cdf1 at large x not close to 1: {cdf1}"
