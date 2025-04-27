
import numpy as np 

from scipy.special import gamma, beta, hyp1f1
from scipy.integrate import quad
from typing import Union, List, Optional

from .hyp_geo import hyp1f1_mellin_transform
from .fcm_dist import fcm2_hat, fcm2_hat_mellin_transform, fcm_sigma, frac_chi_mean, frac_chi2_mean
from .mellin import pdf_by_mellin

# classic result
def hyp2f1_mellin_transform(s, a: float, b: float, c: float):
    C = gamma(c) / gamma(a) / gamma(b)
    return C * gamma(s) * gamma(a - s) * gamma(b - s) / gamma(c - s)


# ----------------------------------------------------------------
# hyp2f1 
class Frac_Hyp2f1:
    def __init__(self, a: float, b: float, c: float, eps: float):
        # combines the Kummer function and FCM2
        self.a = a
        self.b = b
        self.c = c
        self.eps = eps
    
    @property
    def k(self):
        return 2.0 * self.b - 1.0
        
    def integral(self, x):
        assert isinstance(x, float)
        assert x <= 0, f"ERROR: x must be negative, but got {x}"

        k = self.k # this is b
        const = beta(k/2, 0.5) / gamma(0.5)
        fcm2 = fcm2_hat(alpha=1/self.eps, k=k)
        
        def _integrand(s):
            kummer = hyp1f1(self.a, self.c, x*s)
            return const * np.sqrt(s) * kummer * fcm2.pdf(s)  # type: ignore

        p = quad(_integrand, 0.0, np.inf, limit=100000)[0]
        return p

    def integral_by_mellin(self, x):
        assert isinstance(x, float)
        assert x < 0, f"ERROR: x must be negative, but got {x}"
        assert self.a > 0
        c = self.a / 2  # c must be between 0 and self.a
        return pdf_by_mellin(-x, lambda s: self.mellin_transform(s), c=c)

    def mellin_transform(self, s):
        # maps to -x
        k = self.k
        const = beta(k/2, 0.5) / gamma(0.5)
        return const * hyp1f1_mellin_transform(s, self.a, self.c) * self.fcm2_hat_mellin_transform(1.5 - s)

    def fcm2_hat_mellin_transform(self, s):
        return fcm2_hat_mellin_transform(s, alpha=1/self.eps, k=self.k)

    def mellin_transform_expanded(self, s):
        # should be equal to self.mellin_transform(s)
        k = self.k
        const = beta(k/2, 0.5) / gamma(0.5) * 2**(2*s-1)
        kummer = gamma(self.c) / gamma(self.a) * gamma(s) * gamma(self.a-s) / gamma(self.c-s)
        fcm2_hat = gamma((k-1)/2) / gamma( self.eps * (k-1)) * gamma(2*self.eps * (k/2-s)) / gamma(k/2-s) 
        return const * kummer * fcm2_hat
    
    # -----------------------------------------------------
    # Section 5.2.3
    def scaled_integral(self, x):
        k = self.k
        sigma = fcm_sigma(1/self.eps, k)
        Q = 4 * sigma**2 
        B = beta(k/2, 0.5) / gamma(0.5)
        return np.sqrt(Q) / B * self.integral(Q * x)
    
    def scaled_integral_by_fcm2(self, x):
        assert isinstance(x, float)
        assert x <= 0, f"ERROR: x must be negative, but got {x}"

        k = self.k # this is b
        fcm2 = frac_chi2_mean(alpha=1/self.eps, k=k)
        def _integrand(s):
            kummer = hyp1f1(self.a, self.c, x * s)
            return np.sqrt(s) * kummer * fcm2.pdf(s)  # type: ignore

        p = quad(_integrand, 0.0, np.inf, limit=100000)[0]
        return p

    def scaled_integral_by_fcm(self, x):
        assert isinstance(x, float)
        assert x <= 0, f"ERROR: x must be negative, but got {x}"

        k = self.k # this is b
        fcm = frac_chi_mean(alpha=1/self.eps, k=k)
        def _integrand(s):
            kummer = hyp1f1(self.a, self.c, x * s**2)
            return s * kummer * fcm.pdf(s)  # type: ignore

        p = quad(_integrand, 0.0, np.inf, limit=100000)[0]
        return p


def frac_hyp2f1_by_alpha_k(alpha, k, a, c) -> Frac_Hyp2f1:
    return Frac_Hyp2f1(a=a, b=(k+1.0)/2, c=c, eps=1.0/alpha)
