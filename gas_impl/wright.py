import numpy as np 
import mpmath as mp
from scipy.special import gamma
from typing import Union, List


# --------------------------------------------------------------------------------
# this handles negative z using reflection rule
def mp_gamma(z):
    if z > 0: return mp.gamma(z) 
    if z < 0: return mp.pi / mp.gamma(1-z) / mp.sin(z * mp.pi)
    return mp.inf


# --------------------------------------------------------------------------------
def wright1(n: int, x: float, lam: float, mu: float) -> float:
        # the n-th term of Wright Fn
        return np.power(x, n) / gamma(n+1.0) / gamma(lam*n + mu)  # type: ignore


def wright_fn(x: Union[float, int, List], lam: float, mu: float, max_n: int=40, start: int=0):
    if isinstance(x, int):
        x = 1.0 * x
    if isinstance(x, float):
        p = np.array([wright1(k, x, lam, mu) for k in range(start, max_n+1)])
        return sum(p)

    if len(x) >= 1:
        return [wright_fn(x1, lam, mu, max_n=max_n, start=start) for x1 in x]
    raise Exception(f"ERROR: unknown x: {x}")


def wright_m_fn(x, alpha):
    # bad convergence
    return 1.0/(alpha*x) * wright_fn(-x, -alpha, 0)


def mainardi_wright_fn(x, alpha, max_n: int=80):
    # good convergence
    # but this might need mp version to do a really good job
    return wright_fn(-x, -alpha, 1.0-alpha, max_n=max_n)


def mainardi_wright_fn_slope(x, alpha, max_n: int=80):
    # good convergence
    # but this might need mp version to do a really good job
    return -1*wright_fn(-x, -alpha, 1.0-2*alpha, max_n=max_n)


def wright_f_fn(x, alpha):
    # bad convergence
    return wright_fn(-x, -alpha, 0)


def wright_q_fn(x, alpha):
    f = wright_f_fn(x, alpha)
    return wright_fn(-x, -alpha, -1.0) / (-1.0 * f)  # type: ignore

 
def mittag_leffler_fn(x: Union[float, int, List], alpha: float, beta: float=1.0, max_n: int=40, start: int=0):
    
    def _mlf_item(k):
        return x**k / gamma(alpha*k + beta)
    
    if isinstance(x, int):
        x = 1.0 * x
    if isinstance(x, float):
        p = np.array([_mlf_item(k) for k in range(start, max_n+1)])
        return sum(p)

    if len(x) >= 1:
        return [mittag_leffler_fn(x1, alpha, beta, max_n=max_n, start=start) for x1 in x]
    raise Exception(f"ERROR: unknown x: {x}")
