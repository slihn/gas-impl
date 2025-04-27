
import numpy as np
from scipy.special import gamma, poch
from scipy.integrate import quad
from typing import Union, List, Optional

from .wright import wright_fn, mainardi_wright_fn, wright_m_fn_by_levy


# classic result
def hyp1f1_mellin_transform(s, a: float, b: float):
    C = gamma(b) / gamma(a)
    return C * gamma(s) * gamma(a - s) / gamma(b - s)


# ----------------------------------------------------------------
class Frac_Hyp1f1:
    def __init__(self, lam: float, mu: float, a: float, b: float):
        # combines Wright and Kummer functions
        self.lam = lam
        self.mu = mu
        self.a = a
        self.b = b
    
    def series(self, x, max_n: int=80, start: int=0):
        if isinstance(x, int):  x = 1.0 * x
        assert isinstance(x, float)
        a = self.a
        b = self.b

        def wright1(n, x):
            g = 1.0 / gamma(self.mu + self.lam * n)
            pc = poch(a,n) / poch(b,n) if a != b else 1.0
            return np.power(x, n) / gamma(n+1) * g * pc

        p = np.array([wright1(k, x) for k in range(start, max_n+1)])
        return sum(p) 
    
    def integral(self, x):
        if isinstance(x, int):  x = 1.0 * x
        assert isinstance(x, float)
        a = self.a
        b = self.b
        assert a != b
        
        g = gamma(b) / gamma(a) / gamma(b-a)
        def _integrand(t):
            q1 = t**(a-1) if a != 1 else 1.0 
            q2 = (1-t)**(b-a-1) if b-a != 1 else 1.0
            return g * q1 * q2 * wright_fn(x*t, lam=self.lam, mu=self.mu)  # type: ignore

        p = quad(_integrand, 0.0, 1.0, limit=100000)[0]
        return p
    
    def mellin_transform(self, s):
        # maps to -x
        return hyp1f1_mellin_transform(s, self.a, self.b) / gamma(self.mu - self.lam * s)


class Frac_Hyp1f1_M(Frac_Hyp1f1):
    def __init__(self, nu: float, a: float, b: float):
        super().__init__(lam=-nu, mu=1.0-nu, a=a, b=b)

class Frac_Hyp1f1_F(Frac_Hyp1f1):
    def __init__(self, nu: float, a: float, b: float):
        super().__init__(lam=-nu, mu=0.0, a=a, b=b)


# ----------------------------------------------------------------
def frac_hyp1f1_m(x: Union[float, int, List], alpha: float, b: float, c: float, max_n: int=80, start: int=0):
    # TODO legacy, verify this
    if isinstance(x, int):
        x = 1.0 * x

    def wright1(n, x):
        g = gamma(alpha * (n+1)) * np.sin(alpha * (n+1) * np.pi) / np.pi if alpha != 0 else 1.0
        pc = poch(b,n) / poch(c,n) if b != c else 1.0
        return np.power(x, n) / gamma(n+1) * g * pc

    if isinstance(x, float):
        p = np.array([wright1(k, x) for k in range(start, max_n+1)])
        return sum(p) 

    if len(x) >= 1:
        return [frac_hyp1f1_m(x1, alpha, b, c, max_n=max_n, start=start) for x1 in x]
    raise Exception(f"ERROR: unknown x: {x}")


def frac_hyp1f1_m_int(x: Union[float, int, List], alpha: float, b: float, c: float, by_levy=True):
    # TODO legacy, verify this
    assert b != c
    if isinstance(x, int):  x = 1.0 * x

    def _m_wright(x):
        if not by_levy:
            return mainardi_wright_fn(x, alpha)  
        else:
            return wright_m_fn_by_levy(x, alpha)
        
    if isinstance(x, float):
        g = gamma(c) / gamma(b) / gamma(c-b)
        def fn1(t):
            q1 = t**(b-1) if b != 1 else 1.0 
            q2 = (1-t)**(c-b-1) if c-b != 1 else 1.0
            return g * q1 * q2 * _m_wright(-x*t)  # type: ignore

        p = quad(fn1, 0.0, 1.0, limit=1000)[0]
        return p

    if len(x) >= 1:
        return [frac_hyp1f1_m_int(x1, alpha, b, c) for x1 in x]
    raise Exception(f"ERROR: unknown x: {x}")

