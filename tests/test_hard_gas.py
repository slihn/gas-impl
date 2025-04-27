
# test gsas

import numpy as np
import pandas as pd
from scipy.special import gamma
from scipy.integrate import quad
from scipy.stats import levy_stable

from .stable_count_dist import stable_count, gen_stable_count
from .gas_dist import gsas, lihn_stable, gsas_moment
from .gexppow_dist import gexppow, gexppow_moment, gexppow_kurtosis
from .unit_test_utils import *


# -------------------------------------------------------------------------------------
# very hard integrals here, will take a long time

# -------------------------------------------------------------------------------------
# def test_gas_m1():



# -------------------------------------------------------------------------------------
def test_gsas_m2():
    alpha = 1.5
    k = 3.0
    n = 2
    g = gsas(alpha=alpha, k=k)
    m1 = g.moment(n)
    m3 = gsas_moment(n=n, alpha=alpha, k=k)
    delta_precise_up_to(m1, m3, abstol=0.001, reltol=0.001)

    def fn(x):
        return x**n * g.pdf(x) * 2

    m2a = quad(fn, a=0, b=50.0, limit=4000)[0] 
    m2b = quad(fn, a=50.0, b=1000.0, limit=4000)[0] 
    m2 = m2a + m2b
    # 5
    delta_precise_up_to(m1, m2, abstol=0.001, reltol=0.001)

def test_gsas_m4():
    alpha = 1.55  # smaller alpha, harder to converge, be careful
    k = 5.0
    n = 4
    g = gsas(alpha=alpha, k=k)
    m1 = g.moment(n)
    m3 = gsas_moment(n=n, alpha=alpha, k=k)
    delta_precise_up_to(m1, m3, abstol=0.1, reltol=0.001)

    def fn(x):
        return x**n * g.pdf(x) * 2

    m2a = quad(fn, a=0, b=50.0, limit=4000)[0] 
    m2b = quad(fn, a=50.0, b=1000.0, limit=4000)[0] 
    m2c = quad(fn, a=1000.0, b=2000.0, limit=4000)[0] 
    m2 = m2a + m2b + m2c
    # 164
    delta_precise_up_to(m1, m2, abstol=0.1, reltol=0.001)


class Test_GExpPow_Moment:
    alpha = 0.85
    k = 2.1
    g = gexppow(alpha=alpha, k=k)

    def test_m2(self):
        # this takes time
        m2 = self.g.moment(2.0)
        m2a = gexppow_moment(2.0, alpha=self.alpha, k=self.k)
        delta_precise_up_to(m2, m2a)

        def fn(x: float):
            return x**2 *  self.g.pdf(x)
        
        m2i = quad(fn, a=0.0, b=40.0, limit=10000)[0] * 2.0 
        delta_precise_up_to(m2, m2i)

    def test_m4(self):
        # this takes time
        m4 = self.g.moment(4.0)
        m4a = gexppow_moment(4.0, alpha=self.alpha, k=self.k)
        delta_precise_up_to(m4, m4a)

        def fn(x: float):
            return x**4 *  self.g.pdf(x) 
        
        m4i = quad(fn, a=0.0, b=40.0, limit=10000)[0] * 2.0 
        delta_precise_up_to(m4, m4i, abstol=0.001, reltol=0.001)


