import concurrent.futures as futures
from scipy.integrate import quad  # type: ignore
import numpy as np 
import pandas as pd
import mpmath as mp


two_pi_i = 2j * np.pi


def _mellin_integral_result(sn, result):
    ds = (sn[2]-sn[1]) * 1j
    return result * ds / two_pi_i


def _mellin_integral_for_loop(sn, _fn):
    A = 0
    for i in range(len(sn)):  A += _fn(sn[i])
    return _mellin_integral_result(sn, A)


    
def mellin_integral(fn, c=0.0, smax=200.0, num=1000001) -> complex:
    sn = np.linspace(-smax, smax, num=int(num))
    _fn = lambda y: fn(y * 1j + c)
    return _mellin_integral_for_loop(sn, _fn)


# this is too slow for some reason
def mellin_integral_mp(fn, c=0.0, smax=100.0, num=100001) -> mp.mpc:
    c = mp.mpf(c)
    sn = mp.linspace(-smax, smax, int(num))
    _fn = lambda y: fn(y * 1j + c)
    return _mellin_integral_for_loop(sn, _fn)


def mellin_integral_mpr(fn, c=0.0, smax=200.0, num=1000001) -> complex:
    # this uses multiprocess to speed up, still complex calculation
    sn = np.linspace(-smax, smax, num=int(num))

    df = pd.DataFrame(data={'s': sn})
    df['A'] = df.parallel_apply(lambda row: fn(row.s * 1j + c), axis=1)  # type: ignore
    A = df['A'].sum()

    return _mellin_integral_result(sn, A)


def mellin_integral_quad(fn, c=0.0, smax=200.0) -> complex:
    # this is much faster!
    _fn = lambda y: fn(y * 1j + c) 

    def _complex_quad(a, b, limit=10000) -> complex:
        x = quad(lambda x: np.real(_fn(x)), a=a, b=b, limit=limit)[0]  # type: ignore
        y = quad(lambda x: np.imag(_fn(x)), a=a, b=b, limit=limit)[0]  # type: ignore
        return (x + y * 1j)

    A = _complex_quad(a = 0.0,   b = smax)
    B = _complex_quad(a = -smax, b = 0.0) 
    return (A + B) * 1j / two_pi_i


def mellin_integral_quad_mp(fn, c=0.0, smax=200.0):
    c = mp.mpf(c)
    _fn = lambda y: fn(y * 1j + c) 

    def _complex_quad(a, b, limit=10000) -> complex:
        p1 = mp.quad(lambda x: mp.re(_fn(x)), [a, b], limit=limit)
        p2 = mp.quad(lambda x: mp.im(_fn(x)), [a, b], limit=limit)
        return (float(p1) + float(p2) * 1j)  # type: ignore

    A = _complex_quad(a = mp.mpf(0.0),   b = mp.mpf(smax))
    B = _complex_quad(a = mp.mpf(-smax), b = mp.mpf(0.0)) 
    two_pi_i = 2j * mp.pi
    return (A + B) * 1j / two_pi_i


# we name it this way because the imaginary part of the result should be zero for a PDF
# the defualt works well for Wright related functions and alpha-stable distributions
def pdf_by_mellin(x, mellin_transform, c: float = 0.3, 
                  smax: float = 200.0, num: int = 1000001, imag_tol: float = 1e-8, 
                  use_quad: bool = True) -> float:
    assert isinstance(x, float)
    if x == 0: return np.NAN 
    def _integrand(s):
        return mellin_transform(s) * x**(-s) 

    # it is user's job to assert c's range
    if use_quad == True:
        p = mellin_integral_quad(_integrand, c=c, smax=smax) 
    else:
        p = mellin_integral(_integrand, c=c, smax=smax, num=num) 
        
    assert np.imag(p) < imag_tol, f"ERROR: imag part of p is not zero: {np.imag(p)}"  # type: ignore
    return np.real(p)  # type: ignore
