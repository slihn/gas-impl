import numpy as np 
import pandas as pd
from typing import Union, Optional, List
from scipy.special import erf, gamma
from scipy.stats import rv_continuous, norm
from scipy.integrate import quad

from .fcm_dist import frac_chi_mean, fcm_moment



# --------------------------------------------------------------------------------
# exponential power distribution is implemented because it is wrong in scipy

class exppow_gen(rv_continuous):
    def _pdf(self, x, alpha, *args, **kwargs):
        c = 0.5 / gamma(1.0/alpha + 1)
        return c * np.exp(-abs(x)**alpha)

exppow = exppow_gen(name="exponential power", shapes="alpha")


# --------------------------------------------------------------------------------
def gexppow_moment(n: float, alpha: float, k: float):
    n = float(n)
    assert float(int(n)) == n
    if n % 2 != 0: return 0.0  # odd moments are zero
    return norm().moment(int(n)) * fcm_moment(n+1.0, alpha=alpha, k=k) / fcm_moment(1.0, alpha=alpha, k=k)


def gexppow_kurtosis(alpha: Union[float, List], k: float, fisher: bool=True):
    # TODO should have an analytic piece too
    if not isinstance(alpha, float):
        ans = [gexppow_kurtosis(alpha1, k, fisher=fisher) for alpha1 in alpha]
        return ans[0] if len(ans) == 1 else ans

    k = float(k)
    def _fcm_moment(n):
        return fcm_moment(n, alpha=alpha, k=k)

    kurt = 3.0 * _fcm_moment(1.0) * _fcm_moment(5.0) / _fcm_moment(3.0)**2
    return (kurt - 3.0) if fisher else kurt


def gexppow_pdf_at_zero(alpha: float, k: float):
    return 1/np.sqrt(2*np.pi)/fcm_moment(n=1.0, alpha=alpha, k=k)

def gexppow_std_pdf_at_zero(alpha: float, k: float):
    # cast(float) to get implicit complex conversion, np.float64 won't do it
    # https://stackoverflow.com/questions/58599248/python-loop-doesnt-return-complex-numbers-instead-returns-nan
    var = gexppow_moment(n=2.0, alpha=alpha, k=k)
    p0 = gexppow_pdf_at_zero(alpha, k)
    p2 = p0**2 * var
    # print('gexppow_std_pdf_at_zero: debug', [alpha, k, var, p0, p2])
    if isinstance(p2, complex): # complex region, k < 0, only for analytic continuity purpose
        if abs(p2.imag) < 1e-6:
            p = np.sqrt(p2.real)
            # print('gexppow_std_pdf_at_zero (complex):', [alpha, k, p])
            return p
        else: 
            return np.NaN
    else:
        assert isinstance(p2, float)
        p = np.sqrt(p2)
        # print('gexppow_std_pdf_at_zero (normal):', [alpha, k, p])
        return p


class gexppow_gen(rv_continuous):

    def _pdf(self, x, alpha, k, *args, **kwargs):
        # handle array form
        if not isinstance(alpha, float):
            assert len(alpha) == len(x), f"ERROR: len of alpha and x"
            if len(x) == 1:  # trvial case
                return self._pdf(x[0], alpha=alpha[0], k=k[0])
            
            df = pd.DataFrame(data={'x': x, 'alpha': alpha, 'k': k})
            df['pdf'] = df.parallel_apply(lambda row: self._pdf(row['x'], alpha=row['alpha'], k=row['k']), axis=1)  # type: ignore
            return df['pdf'].tolist()

        # integral form
        assert isinstance(x, float)
        assert isinstance(alpha, float)
        assert isinstance(k, float)

        fcm = frac_chi_mean(alpha, k)

        def _kernel(s: float):
            return norm().pdf(x/s) *  fcm.pdf(s)  # type: ignore

        return quad(_kernel, a=0.0, b=np.inf, limit=10000)[0] / fcm.moment(1.0)

    def _cdf(self, x, alpha, k, *args, **kwargs):
        # handle array form
        if not isinstance(alpha, float):
            assert len(alpha) == len(x), f"ERROR: len of alpha and x"
            if len(x) == 1:  # trvial case
                return self._cdf(x[0], alpha=alpha[0], k=k[0])
            
            df = pd.DataFrame(data={'x': x, 'alpha': alpha, 'k': k})
            df['cdf'] = df.parallel_apply(lambda row: self._cdf(row['x'], alpha=row['alpha'], k=row['k']), axis=1)  # type: ignore
            return df['cdf'].tolist()

        # integral form
        assert isinstance(x, float)
        assert isinstance(alpha, float)
        assert isinstance(k, float)

        fcm = frac_chi_mean(alpha, k)
        def _kernel(s: float):
            return s * erf(x/s/np.sqrt(2)) * fcm.pdf(s)  # type: ignore

        cdf1 = quad(_kernel, a=0.0, b=np.inf, limit=10000)[0] / fcm.moment(1.0)
        return cdf1*0.5 + 0.5

    def _argcheck(self, *args, **kwargs):
        # Customize the argument checking here
        alpha = args[0]
        k = args[1]
        return (
            alpha >= 0  # Allow alpha to be zero or positive
            and k >= 0  # I am not very sure about this?
        )

    def _munp(self, n, alpha, k, *args, **kwargs):
        n = float(n)
        alpha = float(alpha)
        k = float(k)
        return gexppow_moment(n, alpha, k)


gexppow = gexppow_gen(name="generalized exponential power", shapes="alpha, k")

