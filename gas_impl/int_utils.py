
import numpy as np 
import pandas as pd
import multiprocessing
from typing import Union, Optional

from scipy.integrate import quad, dblquad


# put integration untilities here
# to test moments and marginals by brute force
# --------------------------------------------------------------------------------


def marginal_from_2d_pdf(pdf_2d, x: float, n: int, 
                         a=-np.inf, b=np.inf, epsabs=1e-3, epsrel=1e-3) -> float:
    assert n in [0, 1]
    x1 = x
    def _integrand_2d(x2):
        x = np.array([x1, x2]) if n == 0 else np.array([x2, x1])
        return pdf_2d(x)
    return quad(_integrand_2d, a, b, epsabs=epsabs, epsrel=epsrel)[0]  # type: ignore


def marginal_from_3d_pdf(pdf_3d, x: float, n: int, 
                         a=-np.inf, b=np.inf, c=-np.inf, d=np.inf,
                         epsabs=1e-3, epsrel=1e-3) -> float:
    assert n in [0, 1, 2]
    x1 = x
    def _integrand_3d(x2, x3):
        x = np.insert(np.array([x2, x3]), n, x1)  # type: ignore
        return pdf_3d(x)
    return dblquad(_integrand_3d, a, b, c, d, epsabs=epsabs, epsrel=epsrel)[0]


# --------------------------------------------------------------------------------
class PDF_Marginal_Integration:
    def __init__(self, dist, x, n, epsabs=1e-3, epsrel=1e-3):
        self.dist = dist
        self.dim = self.dist.n  # dimension of the distribution
        self.x = x
        self.n = n  # axis of the marginal, 0-based
        self.epsabs = epsabs
        self.epsrel = epsrel
        self.queue = multiprocessing.Queue()
        self.slices = 16 if self.dim == 3 else 4

        assert self.n in range(self.dim)
        assert self.dim in [2, 3]

    def calc(self, a=-np.inf, b=np.inf, c=-np.inf, d=np.inf) -> float:
        x = self.x
        n = self.n

        if hasattr(self.dist, 'pdf1'):
            fn = self.dist.pdf1
        elif hasattr(self.dist, '_pdf1'):
            fn = self.dist._pdf1
        else:
            raise Exception("ERROR: dist does not have pdf1 method to use")

        if self.dim == 2:
            return marginal_from_2d_pdf(fn, x, n, a, b, epsabs=self.epsabs, epsrel=self.epsrel)
        elif self.dim == 3:
            return marginal_from_3d_pdf(fn, x, n, a, b, c, d, epsabs=self.epsabs, epsrel=self.epsrel)
        else:
            raise Exception(f"ERROR: marginal_pdf for more than {self.dim} dimension dimension is not supported")

    def calc_slice(self, a) -> float:
        if self.dim == 2:
            if self.slices == 2:
                return self.calc_slice_single_v2(a)
            elif self.slices == 4:
                return self.calc_slice_single_v4(a)
        elif self.dim == 3:
            if self.slices == 4:
                return self.calc_slice_dbl_v4(a)
            elif self.slices == 16:
                return self.calc_slice_dbl_v16(a)
        raise Exception(f"ERROR: invalid slice: {a} for dim={self.dim}")

    def calc_slice_dbl_v16(self, a) -> float:
        assert self.slices == 16
        assert a in range(1, self.slices+1)
        i, j = [ (a-1) // 4, (a-1) % 4 ]
        x = 1.0
        combo = np.array([
            [0.0, x], [x, np.inf],  
            [-np.inf, -x], [-x, 0.0],
        ])
        p = self.calc(combo[i,0], combo[i,1], combo[j,0], combo[j,1])
        self.queue.put(p)
        return p

    def calc_slice_dbl_v4(self, a) -> float:
        assert self.slices == 4
        assert a in range(1, self.slices+1)
        combo = np.array([
            [0.0, np.inf], 
            [-np.inf, 0.0], 
        ])
        i, j = [ (a-1) // 2, (a-1) % 2 ]
        p = self.calc(combo[i,0], combo[i,1], combo[j,0], combo[j,1])
        self.queue.put(p)
        return p

    def calc_slice_single_v4(self, a) -> float:
        assert self.slices == 4
        assert a in range(1, self.slices+1)
        x = 1.0
        combo = np.array([
            [0.0, x], [x, np.inf],  
            [-np.inf, -x], [-x, 0.0],
        ])
        i = a-1
        p = self.calc(combo[i,0], combo[i,1])
        self.queue.put(p)
        return p

    def calc_slice_single_v2(self, a) -> float:
        assert self.slices == 2
        assert a in range(1, self.slices+1)
        combo = np.array([
            [0.0, np.inf], 
            [-np.inf, 0.0], 
        ])
        i = a-1
        p = self.calc(combo[i,0], combo[i,1])
        self.queue.put(p)
        return p


# --------------------------------------------------------------------------------
class PDF_2D_Integration:
    def __init__(self, dist, p1, p2, integrand_x=None, 
                 use_cdf_integrand=False, use_moment_integrand=False, one_sided=False, epsabs=1e-3, epsrel=1e-3):
        self.dist = dist
        self.p1 = p1
        self.p2 = p2
        self.integrand_x = integrand_x  # use _pdf1_integrand if it has value
        self.use_cdf_integrand: bool = use_cdf_integrand  # use the CDF integrand. not the PDF integrand
        self.use_moment_integrand: bool = use_moment_integrand
        self.one_sided: bool = one_sided
        self.epsabs = epsabs
        self.epsrel = epsrel
        self.queue = multiprocessing.Queue()
        self.slices = 16

    def calc(self, a=-np.inf, b=np.inf, c=-np.inf, d=np.inf):
        p1 = self.p1
        p2 = self.p2
        
        # --------------------------------------------------
        if self.integrand_x is not None:
            if self.use_cdf_integrand:
                fn = self.dist._cdf1_integrand
            elif self.use_moment_integrand:
                fn = self.dist._moment_integrand
            else:
                fn = self.dist._pdf1_integrand 

            def _adp_integrand(s1, s2):
                s = np.array([s1, s2])
                return s1**p1 * s2**p2 * fn(self.integrand_x, s)

            return dblquad(
                _adp_integrand, a, b, c, d,
                epsabs=self.epsabs,  # Allow larger absolute error
                epsrel=self.epsrel)[0]

        # --------------------------------------------------
        if hasattr(self.dist, 'pdf1'):
            fn = self.dist.pdf1
        elif hasattr(self.dist, '_pdf1'):
            fn = self.dist._pdf1
        else:
            raise Exception("ERROR: dist does not have pdf1 method to use")

        def _integrand(x1, x2):
            x = np.array([x1, x2])
            return x1**p1 * x2**p2 * fn(x)

        return dblquad(
            _integrand, a, b, c, d,
            epsabs=self.epsabs,  # Allow larger absolute error
            epsrel=self.epsrel)[0]

    def calc_slice(self, a) -> float:
        if self.slices == 4:
            return self.calc_slice_v4(a)
        elif self.slices == 16:
            return self.calc_slice_v16(a)
        else:
            raise Exception("ERROR: invalid slices")

    def calc_slice_v16(self, a) -> float:
        assert self.slices == 16
        assert a in range(1, self.slices+1)
        i, j = [ (a-1) // 4, (a-1) % 4 ]
        x = 1.0
        if not self.one_sided:
            combo = np.array([
                [0.0, x], [x, np.inf],  
                [-np.inf, -x], [-x, 0.0],
            ])
        else:
            x2 = 2.0
            x3 = 3.0
            combo = np.array([
                [0.0, x], [x, x2], [x2, x3], [x3, np.inf],
            ])
            
        p = self.calc(combo[i,0], combo[i,1], combo[j,0], combo[j,1])
        self.queue.put(p)
        return p

    def calc_slice_v4(self, a) -> float:
        assert self.slices == 4
        assert a in range(1, self.slices+1)
        if not self.one_sided:
            combo = np.array([
                [0.0, np.inf], [-np.inf, 0.0], 
            ])
        else:
            x = 1.0
            combo = np.array([
                [0.0, x], [x, np.inf] 
            ])
        i, j = [ (a-1) // 2, (a-1) % 2 ]
        p = self.calc(combo[i,0], combo[i,1], combo[j,0], combo[j,1])
        self.queue.put(p)
        return p

