
from scipy.stats import skewnorm 

from .unit_test_utils import *

from .gas_sn_dist import GAS_SN_Std, GAS_SN_TAIL, gas_sn


# these are slow
class Test_Moments:
    alpha = 1.5
    k = 4.8 
    beta = 1.2

    g = gas_sn(alpha, k, beta)
    g2 = GAS_SN_Std(alpha, k , beta)
    
    def calc_moment(self, g, n):
        def _fn(x):  return x**n * g.pdf(x)
        return quad(_fn, -np.inf, np.inf, epsabs=1e-2, epsrel=1e-2)[0]
 
    def test_moment_1(self):
        n = 1
        p1 = self.g.moment(n)
        p2 = self.calc_moment(self.g, n)
        delta_precise_up_to(p1, p2)
        
        p3 = self.g2._moment(n)
        delta_precise_up_to(p1, p3)

    def test_moment_2(self):
        n = 2
        p1 = self.g.moment(n)
        p2 = self.calc_moment(self.g, n)
        delta_precise_up_to(p1, p2, abstol=1e-2)

        p3 = self.g2._moment(n)
        delta_precise_up_to(p1, p3)

    def test_moment_3(self):
        n = 3
        p1 = self.g.moment(n)
        p2 = self.calc_moment(self.g, n)
        delta_precise_up_to(p1, p2, abstol=1e-2)

        p3 = self.g2._moment(n)
        delta_precise_up_to(p1, p3)


class Test_GAS_SN_TAIL_Moments:
    alpha = 1.35
    k = 4.1
    beta = 1.2

    g = GAS_SN_TAIL(alpha, k, beta)
    g2 = gas_sn(alpha, k, beta)

    def calc_moment(self, g, n):
        def _fn(x):  return x**n * g.pdf(x)
        return quad(_fn, -np.inf, np.inf, epsabs=1e-2, epsrel=1e-2)[0]

    def test_moment_1(self):
        n = 1
        exact = self.g2.moment(n)
        approx = self.calc_moment(self.g, n)
        delta_precise_up_to(exact, approx, abstol=1e-2, reltol=1e-2)

    def test_moment_2(self):
        n = 2
        exact = self.g2.moment(n)
        approx = self.calc_moment(self.g, n)
        delta_precise_up_to(exact, approx, abstol=1e-2, reltol=1e-2)

    def test_moment_3(self):
        n = 3
        exact = self.g2.moment(n)
        approx = self.calc_moment(self.g, n)
        delta_precise_up_to(exact, approx, abstol=1e-2, reltol=1e-2)
