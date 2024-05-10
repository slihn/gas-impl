import concurrent.futures as futures
from scipy.integrate import quad  # type: ignore
import numpy as np 
import pandas as pd
import mpmath as mp
from numba import njit, prange

# TODO want to use numba to speed things up, but not quite work yet

two_pi_i = 2j * np.pi


def hankel_integral(fn, delta=0.000001j, smin=-2000, num=1000000) -> complex:
    sn = np.linspace(smin, 0, num=int(num))
    ds = sn[2]-sn[1]

    A = B = 0
    for i in prange(len(sn)):
        A += fn(sn[i] - delta)
        B += fn(sn[i] + delta)

    return (A-B) * ds / two_pi_i


# this is too slow for some reason
def hankel_integral_mp(fn, delta=0.000001j, smin=-2000, num=1000000) -> mp.mpc:
    sn = mp.linspace(smin, 0, int(num))
    ds = sn[2]-sn[1]

    A = B = 0
    for i in prange(len(sn)):
        A += fn(sn[i] - delta)
        B += fn(sn[i] + delta)

    return (A-B) * ds / two_pi_i


def hankel_integral_mpr(fn, delta=0.000001j, smin=-2000, num=1000000) -> complex:
    sn = np.linspace(smin, 0, int(num))
    ds = sn[2]-sn[1]

    df = pd.DataFrame(data={'s': sn})
    df['A'] = df.parallel_apply(lambda row: fn(row.s - delta), axis=1)  # type: ignore
    df['B'] = df.parallel_apply(lambda row: fn(row.s + delta), axis=1)  # type: ignore
    A = df['A'].sum()
    B = df['B'].sum()
    return (A-B) * ds / two_pi_i


def complex_quad(fn, a=0, b=np.inf) -> complex:
    x = quad(lambda x: np.real(fn(x)), a=a, b=b, limit=1000)[0] 
    y = quad(lambda x: np.imag(fn(x)), a=a, b=b, limit=1000)[0] 
    return (x + y * 1j)
