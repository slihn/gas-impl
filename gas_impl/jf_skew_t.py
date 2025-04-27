#
# Copy from SciPy 1.14.1. My scipy version is too behind, but I need this
#

import numpy as np
import scipy.special as sc

from scipy._lib._util import _lazywhere
from scipy.stats._distn_infrastructure import rv_continuous, _ShapeInfo
from scipy.stats import beta


class jf_skew_t_gen(rv_continuous):
    r"""Jones and Faddy skew-t distribution.

    %(before_notes)s

    Notes
    -----
    The probability density function for `jf_skew_t` is:

    .. math::

        f(x; a, b) = C_{a,b}^{-1}
                    \left(1+\frac{x}{\left(a+b+x^2\right)^{1/2}}\right)^{a+1/2}
                    \left(1-\frac{x}{\left(a+b+x^2\right)^{1/2}}\right)^{b+1/2}

    for real numbers :math:`a>0` and :math:`b>0`, where
    :math:`C_{a,b} = 2^{a+b-1}B(a,b)(a+b)^{1/2}`, and :math:`B` denotes the
    beta function (`scipy.special.beta`).

    When :math:`a<b`, the distribution is negatively skewed, and when
    :math:`a>b`, the distribution is positively skewed. If :math:`a=b`, then
    we recover the `t` distribution with :math:`2a` degrees of freedom.

    `jf_skew_t` takes :math:`a` and :math:`b` as shape parameters.

    %(after_notes)s

    References
    ----------
    .. [1] M.C. Jones and M.J. Faddy. "A skew extension of the t distribution,
           with applications" *Journal of the Royal Statistical Society*.
           Series B (Statistical Methodology) 65, no. 1 (2003): 159-174.
           :doi:`10.1111/1467-9868.00378`

    %(example)s

    """
    def _shape_info(self):
        ia = _ShapeInfo("a", False, (0, np.inf), (False, False))
        ib = _ShapeInfo("b", False, (0, np.inf), (False, False))
        return [ia, ib]

    def _pdf(self, x, a, b):
        c = 2 ** (a + b - 1) * sc.beta(a, b) * np.sqrt(a + b)
        d1 = (1 + x / np.sqrt(a + b + x ** 2)) ** (a + 0.5)
        d2 = (1 - x / np.sqrt(a + b + x ** 2)) ** (b + 0.5)
        return d1 * d2 / c

    def _rvs(self, a, b, size=None, random_state=None):
        d1 = random_state.beta(a, b, size)  # type: ignore
        d2 = (2 * d1 - 1) * np.sqrt(a + b)
        d3 = 2 * np.sqrt(d1 * (1 - d1))
        return d2 / d3

    def _cdf(self, x, a, b):
        y = (1 + x / np.sqrt(a + b + x ** 2)) * 0.5
        return sc.betainc(a, b, y)

    def _ppf(self, q, a, b):
        d1 = beta.ppf(q, a, b)
        d2 = (2 * d1 - 1) * np.sqrt(a + b)
        d3 = 2 * np.sqrt(d1 * (1 - d1))
        return d2 / d3

    def _munp(self, n, a, b):
        """Returns the n-th moment(s) where all the following hold:

        - n >= 0
        - a > n / 2
        - b > n / 2

        The result is np.nan in all other cases.
        """
        def nth_moment(n_k, a_k, b_k):
            """Computes E[T^(n_k)] where T is skew-t distributed with
            parameters a_k and b_k.
            """
            num = (a_k + b_k) ** (0.5 * n_k)
            denom = 2 ** n_k * sc.beta(a_k, b_k)

            indices = np.arange(n_k + 1)
            sgn = np.where(indices % 2 > 0, -1, 1)
            d = sc.beta(a_k + 0.5 * n_k - indices, b_k - 0.5 * n_k + indices)
            sum_terms = sc.comb(n_k, indices) * sgn * d

            return num / denom * sum_terms.sum()

        nth_moment_valid = (a > 0.5 * n) & (b > 0.5 * n) & (n >= 0)
        return _lazywhere(
            nth_moment_valid,
            (n, a, b),
            np.vectorize(nth_moment, otypes=[np.float64]),  # type: ignore
            np.nan,
        )


jf_skew_t = jf_skew_t_gen(name='jf_skew_t')

