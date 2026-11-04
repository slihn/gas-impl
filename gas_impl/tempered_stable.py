"""Tempered stable clocks (TSS, CTS1) and their symmetric distributions (NTSR, NTSR1).

The standard tss clock has Laplace transform
    E exp(-s T) = exp(-c * ((lam + s)**alpha - lam**alpha)),
where 0 < alpha < 1, c > 0 and lam > 0. 

With an independent standard normal Z, ntsr = Z / exp(U), U = -zeta * (T - E[T]), 
uses a centered/reflected log clock, matching the log-tempered-stable approximation to fractional chi.
The explicit branch zeta=+1 or -1 corresponds to positive or negative k (chracteristic branch).

For the exact index-one book log-law, cts1(delta,lam) has mean zero and support on the entire real line.
And ntsr1(delta,lam,zeta) is its normal/exponential-clock ratio. 
Their finite Levy coefficient delta replaces the divergent positive-TSS c.

CTS1 and NTSR1 are designed to mirror log-FCM and GSAS.

Moment conventions
------------------
TSS and CTS1 have explicit cumulants and finite nonnegative integer moments.
Raw integer moments are complete Bell polynomials in the cumulants; density
quadrature is unnecessary. For the TSS Levy measure, delta = c*alpha /
Gamma(1-alpha), so c and the CTS Levy coefficient delta are different scales.

More generally, a centered positive-jump CTS with index a in (0,2) and Levy
measure delta*exp(-lam*x)/x**(1+a) dx has kappa_1=0 and
kappa_n=delta*Gamma(n-a)/lam**(n-a), n>=2. The bilateral CTS cumulant adds
(-1)**n*delta_minus*Gamma(n-a)/lam_minus**(n-a).

Only the index-one CTS is implemented here, since it is the approximation of log-FCM near the lognormal limit. 
At a=1 its cumulants are regular although the generic CGF expression with Gamma(-a) needs its logarithmic limit.

Clock powers and exponential-clock powers are different expectations.
For W=exp(U), E[W**q]=exp(K_U(q)). Hence for R=Z/exp(U), independence gives
E[abs(R)**q] = 2**(q/2)*Gamma((q+1)/2)/sqrt(pi)*exp(K_U(-q)),
when q>-1 and K_U(-q) is finite. 
Arbitrary fractional powers generally need a separate
integral or special-function representation. SciPy moment(n) here exposes
nonnegative integer raw moments, not arbitrary real or absolute moments.

Like lnn, these are scipy.stats.rv_continuous instances: calling
tss(alpha, c, lam) or ntsr(alpha, c, lam, zeta) freezes the shapes;
cts1 and ntsr1 have the same SciPy interface, with shapes (delta,lam)
and (delta,lam,zeta), respectively.
The loc and scale have their ordinary SciPy meanings. 

Density evaluation uses an exponentially tilted Kanter integral for TSS,
and positive normal-mixture quadrature for NTSR. The index-one CTS
extension uses centered characteristic functions and Esscher tilting.
Positive-TSS sampling uses exact exponential
rejection with convolution splitting; its work grows with c * lam**alpha.

Reference: Massing, 'Parametric Estimation of Tempered Stable Laws',
https://arxiv.org/abs/2303.07060, Sections 2.1–2.3.
"""

from functools import lru_cache
import math
from math import comb

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import quad
from scipy.special import gammaln
from scipy.stats import invgauss, levy_stable, norm, rv_continuous
from scipy.stats._distn_infrastructure import rv_continuous_frozen

from .tilted_stable import one_sided_stable, zolotarev_log_A


def tss_cumulant(n, alpha, c, lam):
    """Nth standard TSS cumulant; integer n >= 1, shapes broadcast.

    kappa_n = c*alpha*Gamma(n-alpha)/Gamma(1-alpha)*lam**(alpha-n).
    Thus mean=c*alpha*lam**(alpha-1) and
    variance=c*alpha*(1-alpha)*lam**(alpha-2).
    
    For n>=2, centering leaves the cumulants unchanged and reflection by -zeta multiplies them
    by (-zeta)**n. These are cumulants, not raw moments of orders n>=2.
    No SciPy loc/scale adjustment is included; see tss.moment(n).
    """
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError('n must be a positive integer')
    alpha, c, lam = np.broadcast_arrays(alpha, c, lam)
    return np.exp(np.log(c) + np.log(alpha) + gammaln(n - alpha)
                  - gammaln(1 - alpha) + (alpha - n) * np.log(lam))


def tss_laplace(s, alpha, c, lam):
    """E exp(-s T) in standard form; s >= 0 (no loc/scale adjustment)."""
    s, alpha, c, lam = np.broadcast_arrays(s, alpha, c, lam)
    if np.any(s < 0):
        raise ValueError('s must be nonnegative')
    return np.exp(-c * np.power(lam, alpha)
                  * np.expm1(alpha * np.log1p(s / lam)))


def tss_cf(u, alpha, c, lam):
    """Characteristic function of the standard TSS clock."""
    u, alpha, c, lam = np.broadcast_arrays(u, alpha, c, lam)
    return np.exp(-c * np.power(lam, alpha)
                  * np.expm1(alpha * np.log1p(-1j * u / lam)))


@lru_cache(maxsize=128)
def _tss_stable(alpha) -> rv_continuous_frozen:
    # Reuse the existing positive stable constructor: LT = exp(-s**alpha).
    stable = one_sided_stable(float(alpha))
    assert isinstance(stable, rv_continuous_frozen)
    return stable


@lru_cache(maxsize=16384)
def _tss_logpdf(x, alpha, c, lam):
    # PDF/CDF mixture integrals reuse clock nodes while the return x varies.
    if x <= 0:
        return -np.inf
    if alpha == 0.5:
        return (np.log(c) - np.log(2 * np.sqrt(np.pi)) - 1.5 * np.log(x)
                + c * np.sqrt(lam) - lam * x - c*c / (4*x))
    # Kanter: S=A(Q)*E**(-(1-alpha)/alpha), Q uniform, E exponential.
    # Conditional on Q, the stable density is d/x * exp(v-exp(v)), where
    # d=alpha/(1-alpha), v=d*(log(A(Q))-log(x/c**(1/alpha))).
    # Integrate the positive kernel after subtracting its maximum. Unlike
    # dstable(...)*exp(tilt), this retains the small-x density under strong tilt.
    d = alpha / (1 - alpha)
    log_x = np.log(x) - np.log(c) / alpha
    log_a0 = np.log(alpha) + (1-alpha)/alpha * np.log1p(-alpha)
    v0 = d * (log_a0 - log_x)
    if v0 > 700:
        return -np.inf
    if v0 < 0:
        # The ordinary stable routine is accurate on this side of the Kanter
        # mode. Keep it out of the small-x region where its error is amplified
        # exponentially by tempering, and avoid resolving a far-right angle
        # peak next to q=1 when the clock density is already negligible.
        log_scale = np.log(c) / alpha
        return (_tss_stable(alpha).logpdf(np.exp(log_x))
                - log_scale
                + c*lam**alpha-lam*x)
    height = v0 - np.exp(v0)
    width = 1 / np.sqrt(1 + alpha*np.pi**2*np.expm1(v0))

    def integrand(y):
        q = width*y
        if q >= 1:
            return 0.0
        # Compute log A(q)-log A(0) without subtracting nearly equal logs.
        # Strong tilting can concentrate the angle at q << machine epsilon.
        if q < 1e-2:
            q2 = q*q
            offset = ((1-alpha)*np.pi**2*q2/2
                      + (1-alpha)*(1-alpha+alpha**2)*np.pi**4*q2**2/36
                      + (1-alpha)*(1-2*alpha+3*alpha**2-2*alpha**3+alpha**4)
                      * np.pi**6*q2**3/405)
        else:
            offset = zolotarev_log_A(q, alpha)-log_a0
        dv = d*offset
        with np.errstate(over='ignore'):
            exponent = dv-np.exp(v0)*np.expm1(dv)
        return np.exp(exponent)

    # Expand away from the peak in finite pieces. A single huge interval can
    # miss the peak, whereas zero-extension to infinity adds a discontinuity
    # at the finite angle endpoint. The kernel is decreasing here;
    # once it underflows to zero, subsequent intervals are negligible.
    upper = 1/width
    y = min(1.0, upper)
    bounds = [0.0, y]
    for _ in range(1024):
        if y == upper or integrand(y) == 0:
            break
        y = min(2*y, upper)
        bounds.append(y)
    mass = sum(quad(integrand, lo, hi, epsabs=1e-13, epsrel=2e-10, limit=200)[0]
               for lo, hi in zip(bounds[:-1], bounds[1:]))
    return (np.log(d)-np.log(x) + height + c*lam**alpha-lam*x
            + np.log(width) + np.log(mass))


@lru_cache(maxsize=512)
def _tss_tail(x, alpha, c, lam, upper):
    if x <= 0:
        return 1.0 if upper else 0.0
    if alpha == 0.5:
        dist = invgauss(mu=1 / (c * np.sqrt(lam)), scale=c*c / 2)
        return dist.sf(x) if upper else dist.cdf(x)
    mean = float(tss_cumulant(1, alpha, c, lam))
    # Work in units of the clock mean, and split there so a concentrated clock
    # cannot fall between the nodes of an integral over (0, infinity).
    f = lambda y: np.exp(_tss_logpdf(mean * y, alpha, c, lam)) * mean
    y = x / mean
    if upper:
        bounds = [y, 1.0, np.inf] if y < 1 else [y, np.inf]
    else:
        bounds = [0.0, 1.0, y] if y > 1 else [0.0, y]
    return sum(quad(f, lo, hi, epsabs=0, epsrel=2e-9, limit=250)[0]
               for lo, hi in zip(bounds[:-1], bounds[1:]))


def _tss_moment(n, alpha, c, lam):
    """Raw integer moment from cumulants, using the complete Bell recurrence.

    M_0=1; M_n=sum_{j=1}^n binom(n-1,j-1)*kappa_j*M_{n-j}.
    In particular M_2=kappa_2+kappa_1**2; for a centered clock,
    M_4=kappa_4+3*kappa_2**2. No numerical density integration is used.
    """
    moments = [np.ones(np.broadcast(alpha, c, lam).shape)]
    cumulants = [tss_cumulant(j, alpha, c, lam) for j in range(1, n + 1)]
    for order in range(1, n + 1):
        moment = np.zeros_like(moments[0])
        for j in range(1, order + 1):
            moment += comb(order - 1, j - 1) * cumulants[j - 1] * moments[order - j]
        moments.append(moment)
    return moments[n]


def _tss_sample(alpha, c, lam, size, rng):
    shape = () if size is None else size
    aa, cc, ll = np.broadcast_arrays(np.broadcast_to(alpha, shape),
                                     np.broadcast_to(c, shape), np.broadcast_to(lam, shape))
    out = np.empty(aa.size)
    parameters = np.column_stack([aa.ravel(), cc.ravel(), ll.ravel()])
    unique, group = np.unique(parameters, axis=0, return_inverse=True)
    for g, (a, intensity, rate) in enumerate(unique):
        indices = np.flatnonzero(group == g)
        if a == 0.5:
            out[indices] = rng.wald(intensity / (2 * np.sqrt(rate)), intensity**2 / 2,
                                   size=len(indices))
            continue
        # A single stable proposal is accepted with probability exp(-c*lam**a).
        # Split the clock into independent increments with exponent <= 1, giving
        # acceptance >= exp(-1) per increment without a Gaussian approximation.
        pieces = max(1, int(np.ceil(intensity * rate**a)))
        increment_c = intensity / pieces
        stable_scale = increment_c**(1 / a)
        values = np.zeros(len(indices))
        for _ in range(pieces):
            pending = np.arange(len(indices))
            while pending.size:
                proposed = stable_scale * _tss_stable(a).rvs(size=pending.size, random_state=rng)
                accepted = np.log(rng.uniform(size=pending.size)) <= -rate * proposed
                values[pending[accepted]] += proposed[accepted]
                pending = pending[~accepted]
        out[indices] = values
    return out.reshape(aa.shape)


class tss_gen(rv_continuous):
    """Positive CTS/TSS with shapes (alpha, c, lam) and the Laplace transform above.

    All real power moments exist for the standard positive clock. moment(n)
    evaluates nonnegative integer raw moments analytically from tss_cumulant
    via the complete Bell recurrence. Fractional/negative orders are not
    supplied by that SciPy method. For the standard clock, the MGF is
    exp(c*(lam**alpha-(lam-s)**alpha)), finite for real s<=lam, including
    the endpoint. See tss_centered_cgf for exponential moments after centering.

    The standard law has support (0, infinity); loc adds a
    deterministic shift and scale multiplies the clock. alpha=1/2 is inverse
    Gaussian. Generic SciPy quantiles invert the numerical CDF.
    """

    def _argcheck(self, alpha, c, lam):
        return ((alpha > 0) & (alpha < 1) & (c > 0) & (lam > 0)
                & np.isfinite(alpha) & np.isfinite(c) & np.isfinite(lam))

    def _logpdf(self, x, alpha, c, lam):
        return np.vectorize(_tss_logpdf, otypes=[float])(x, alpha, c, lam)

    def _pdf(self, x, alpha, c, lam):
        return np.exp(self._logpdf(x, alpha, c, lam))

    def _cdf(self, x, alpha, c, lam):
        return np.vectorize(_tss_tail, otypes=[float])(x, alpha, c, lam, False)

    def _sf(self, x, alpha, c, lam):
        return np.vectorize(_tss_tail, otypes=[float])(x, alpha, c, lam, True)

    def _rvs(self, alpha, c, lam, size=None, random_state=None):
        rng = self.random_state if random_state is None else random_state
        return _tss_sample(alpha, c, lam, size, rng)

    def _munp(self, n, alpha, c, lam):
        return _tss_moment(int(n), alpha, c, lam)

    def _stats(self, alpha, c, lam):
        mean = tss_cumulant(1, alpha, c, lam)
        var = tss_cumulant(2, alpha, c, lam)
        skew = tss_cumulant(3, alpha, c, lam) / var**1.5
        kurt = tss_cumulant(4, alpha, c, lam) / var**2
        return mean, var, skew, kurt


tss = tss_gen(a=0.0, name='tss', shapes='alpha, c, lam')


def tss_centered_cgf(s, alpha, c, lam):
    """CGF of T-E[T]; real orders above lam give infinity, complex orders use the principal branch.

    This is the centered *level* clock, not log(T). A power series for the
    centered exponent avoids subtracting its large linear term near LNN.
    With mu=c*alpha*lam**(alpha-1),
        K(s)=c*(lam**alpha-(lam-s)**alpha)-mu*s,
    finite for s<=lam, including s=lam where K(lam)=c*(1-alpha)*lam**alpha.
    For U=-zeta*(T-mu), K_U(q)=K(-zeta*q). Consequently the return
    R=Z/exp(U) uses exp(K(zeta*q)) in its absolute moment of order q.
    """
    s, alpha, c, lam = np.broadcast_arrays(s, alpha, c, lam)
    r = -s / lam
    with np.errstate(divide='ignore', invalid='ignore'):
        residual = np.expm1(alpha * np.log1p(r)) - alpha*r
        # Near alpha=1 the centered exponent is itself O(1-alpha).
        # Factor out that cancellation before multiplying by the divergent c.
        near_one = (alpha > .5) & (r != -1)
        residual = np.where(near_one,
                            (1+r)*np.expm1((alpha-1)*np.log1p(r))+(1-alpha)*r,
                            residual)
        residual = np.where(r == -1, alpha-1, residual)
    small = np.abs(r) < .01
    if np.any(small):
        rr, aa = r[small], alpha[small]
        coefficient = aa*(aa-1)/2
        power = rr*rr
        series = coefficient*power
        for n in range(3, 15):
            coefficient = coefficient*(aa-n+1)/n
            power = power*rr
            series = series+coefficient*power
        residual = np.array(residual, copy=True)
        residual[small] = series
    result = -c*lam**alpha*residual
    if not np.iscomplexobj(s):
        result = np.where(s > lam, np.inf, result)
    return result


@lru_cache(maxsize=512)
def _ntsr_mixture(x, alpha, c, lam, zeta, density):
    mean = float(tss_cumulant(1, alpha, c, lam))
    sd = float(np.sqrt(tss_cumulant(2, alpha, c, lam)))
    if x == 0:
        return (np.exp(tss_centered_cgf(-zeta, alpha, c, lam))/np.sqrt(2*np.pi)
                if density else .5)
    log_x = np.log(abs(x))

    def integrand(y):
        t = mean+sd*y
        if t <= 0:
            return 0.0
        log_vol = zeta*sd*y
        log_arg = log_x-log_vol
        if log_arg > 350:
            return 0.0
        arg = np.exp(log_arg)
        log_weight = _tss_logpdf(t, alpha, c, lam)+np.log(sd)
        conditional = norm.logpdf(arg)-log_vol if density else norm.logsf(arg)
        return np.exp(log_weight+conditional)

    # Integrate in centered clock-SD coordinates; raw-clock quadrature can
    # miss the mass when its diverging mean is much larger than its SD.
    lower = -mean/sd
    bounds = {-np.inf, -2., 0., 2., np.inf}
    if -2 < lower < 2:
        bounds.add(lower)
    peak = zeta*log_x/sd
    if peak > lower:
        bounds.add(peak)
        distance = 4.
        while distance < abs(peak):
            bounds.add(np.sign(peak)*distance)
            distance *= 2
    bounds = sorted(bounds)
    return sum(quad(integrand, lo, hi, epsabs=0, epsrel=3e-8, limit=250)[0]
               for lo, hi in zip(bounds[:-1], bounds[1:]))


class ntsr_gen(rv_continuous):
    """Symmetric normal ratio R = Z / exp(U), with U = -zeta*(S-mu).

    S ~ tss(alpha,c,lam) and Z ~ N(0,1) are independent, and
    mu = E[S] = c*alpha*lam**(alpha-1). S is the positive subordinator;
    U is its centered/reflected log clock. The denominator exp(U) has
    geometric mean one.

    Shapes are 0 < alpha < 1, c > 0, lam > 0, and zeta in {+1,-1}.
    For the FCM/GSAS maps, zeta=+1 selects k>1 and zeta=-1 selects k<0.
    Both branches give a symmetric return distribution. SciPy loc and
    scale act on R: the returned variable is loc + scale*R.

    Let K(s) = log E[exp(s*(S-mu))]. For an even integer n >= 0,
        E[R**n] = (n-1)!! * exp(K(n*zeta)), with (-1)!! = 1.
    For zeta=+1, the absolute moment of positive order p is finite iff
    p <= lam (including the boundary). For zeta=-1 all positive absolute
    moments are finite. Odd integer moments are zero when they exist.

    More generally, E[abs(R)**q] =
        2**(q/2)*Gamma((q+1)/2)/sqrt(pi)*exp(K(q*zeta)),
    for q>-1 and q*zeta<=lam. The factor q>-1 comes from the normal;
    the second condition comes from the clock MGF. For q<=-1 the absolute
    moment diverges. For even n=2j the normal factor is (2j-1)!!.
        Variance = exp(K(2*zeta)) and 
        excess kurtosis = 3*expm1(K(4*zeta)-2*K(2*zeta)), 
    provided the required moments exist.

    For Y=loc+scale*R, E[abs(Y-loc)**q]=scale**q*E[abs(R)**q].
    SciPy moment(n) returns integer signed raw moments; it does not evaluate
    the fractional absolute-moment formula. Symmetry gives an odd moment
    of zero only when its absolute moment is finite.
    """

    def _argcheck(self, alpha, c, lam, zeta):
        return tss._argcheck(alpha, c, lam) & (np.abs(zeta) == 1)

    def _pdf(self, x, alpha, c, lam, zeta):
        return np.vectorize(_ntsr_mixture, otypes=[float])(x, alpha, c, lam, zeta, True)

    def _cdf(self, x, alpha, c, lam, zeta):
        tail = self._sf(np.abs(x), alpha, c, lam, zeta)
        return np.where(x <= 0, tail, 1-tail)

    def _sf(self, x, alpha, c, lam, zeta):
        tail = np.vectorize(_ntsr_mixture, otypes=[float])(x, alpha, c, lam, zeta, False)
        return np.where(x >= 0, tail, 1-tail)

    def _rvs(self, alpha, c, lam, zeta, size=None, random_state=None):
        rng = self.random_state if random_state is None else random_state
        t = tss.rvs(alpha, c, lam, size=size, random_state=rng)
        mean = tss_cumulant(1, alpha, c, lam)
        return rng.standard_normal(size)*np.exp(zeta*(t-mean))

    def _munp(self, n, alpha, c, lam, zeta):
        exists = (zeta < 0) | (n <= lam)
        if n % 2:
            return np.where(exists, 0.0, np.nan)
        order = int(n)//2
        normal_moment = 1 if order == 0 else np.prod(np.arange(1, 2*order, 2, dtype=float))
        return normal_moment*np.exp(tss_centered_cgf(n*zeta, alpha, c, lam))

    def _stats(self, alpha, c, lam, zeta):
        log_m2 = tss_centered_cgf(2*zeta, alpha, c, lam)
        log_m4 = tss_centered_cgf(4*zeta, alpha, c, lam)
        mean = np.where((zeta < 0) | (lam >= 1), 0.0, np.nan)
        var = np.where(np.isnan(mean), np.nan, np.exp(log_m2))
        skew = np.where((zeta < 0) | (lam >= 3), 0.0, np.nan)
        with np.errstate(invalid='ignore'):
            kurt = np.where(np.isfinite(var), 3*np.expm1(log_m4-2*log_m2), np.nan)
        return mean, var, skew, kurt


ntsr = ntsr_gen(name='ntsr', shapes='alpha, c, lam, zeta')


def cts1_cgf(s: ArrayLike, delta: ArrayLike, lam: ArrayLike):
    """Centered positive-jump CTS CGF at index one, with finite Levy coefficient delta.

        K(s)=delta*((lam-s)*log(1-s/lam)+s), finite for real s <= lam.

    The law has mean zero and support R. It is a compensated log clock,
    rather than a positive subordinator. Complex orders use the principal branch.
    At s=lam, the logarithmic term has limit zero and K(lam)=delta*lam.
    exp(K(s)) is E[exp(s*S)], not E[S**s]. For W=exp(-zeta*S),
    E[W**q]=exp(K(-zeta*q)); ntsr1 absolute moments use K(zeta*q).
    Scalar or array-like orders and shapes broadcast to a common shape.
    """
    s,delta,lam=np.broadcast_arrays(s,delta,lam)
    r=-s/lam
    with np.errstate(divide='ignore',invalid='ignore'):
        exponent=(1+r)*np.log1p(r)-r
    exponent=np.where(r == -1,1.,exponent)
    small=abs(r)<.1
    if np.any(small):
        rr=r[small]
        series=np.zeros_like(rr)
        for n in range(22,1,-1):
            series=series*rr+(-1.)**n/(n*(n-1))
        exponent=np.array(exponent,copy=True)
        exponent[small]=rr*rr*series
    result=delta*lam*exponent
    if not np.iscomplexobj(s):
        result=np.where(s>lam,np.inf,result)
    return result


def cts1_cf(u: ArrayLike, delta: ArrayLike, lam: ArrayLike):
    """Characteristic function of the centered index-one positive-jump CTS law."""
    return np.exp(cts1_cgf(1j*np.asarray(u),delta,lam))


def cts1_cumulant(n: int, delta: ArrayLike, lam: ArrayLike):
    """Centered CTS1 cumulants: k1=0, kn=delta*Gamma(n-1)/lam**(n-1), n>=2.

    These follow by integrating x**n against delta*exp(-lam*x)/x**2 dx.
    The first cumulant is fixed by compensation, not by that divergent
    integral. Mean=0, variance=delta/lam, skewness=1/sqrt(delta*lam),
    and excess kurtosis=2/(delta*lam). In particular E[S**3]=delta/lam**2
    and E[S**4]=2*delta/lam**3+3*(delta/lam)**2. cts1.moment(n) converts
    cumulants to raw integer moments; loc/scale adjustments are not included
    in this function. Scalar or array-like delta and lam broadcast.
    """
    if not isinstance(n,(int,np.integer)) or n<1:
        raise ValueError('n must be a positive integer')
    delta,lam=np.broadcast_arrays(delta,lam)
    if n==1:
        return np.zeros(delta.shape)
    return np.exp(np.log(delta)+gammaln(n-1)-(n-1)*np.log(lam))


def _cts1_exponent(y,q):
    # Scalar spectral kernel q*((1-iy)*log(1-iy)+iy).
    if abs(y)<.1:
        r=-1j*y
        series=0j
        for n in range(22,1,-1):
            series=series*r+(-1.)**n/(n*(n-1))
        return q*r*r*series
    angle=math.atan(y)
    log_modulus=.5*math.log1p(y*y)
    return complex(q*(log_modulus-y*angle),q*(y-angle-y*log_modulus))


_CTS1_STABLE: rv_continuous_frozen = rv_continuous_frozen(
    levy_stable, 1., 1., scale=np.pi/2
)
_CTS1_STABLE.dist.parameterization='S1'


@lru_cache(maxsize=16384)
def _cts1_logpdf(x,delta,lam):
    # Exact Esscher saddlepoint: move x to the mean of another CTS1 law.
    # This avoids multiplying an underflowed stable density by a huge tilt.
    r=x/delta
    log_q=math.log(delta)+math.log(lam)
    log_q_star=log_q-r
    if -r>700:
        return -np.inf
    if abs(r)<.01:
        correction=0.
        for n in range(16,1,-1):
            correction=correction*r+(-1.)**n/math.factorial(n)
        correction*=r*r
    else:
        correction=math.expm1(-r)+r
    exponent=-delta*lam*correction
    if not np.isfinite(exponent):
        return -np.inf
    if log_q_star<0:
        # Canonical stable CF exp(-pi*|t|/2-it*log|t|). Here its
        # argument is >= -1, safely away from the problematic far-left tail.
        # SciPy 1.10's alpha=1 logpdf omits the scale/location adjustment
        # made by pdf. The density is safe to form on this side of the mode.
        log_height=(math.exp(log_q_star)
                    +math.log(float(_CTS1_STABLE.pdf(-log_q_star-1))))
    else:
        root_q=math.exp(log_q_star/2)
        q=math.exp(log_q_star)

        def integrand(t):
            w=_cts1_exponent(t/root_q,q)
            return math.exp(w.real)*math.cos(w.imag)

        height=quad(integrand,0.,40.,epsabs=2e-12,epsrel=2e-11,limit=300)[0]/np.pi
        if height<=0:
            raise ArithmeticError('index-one CTS density inversion lost positivity')
        log_height=log_q_star/2+math.log(height)
    return exponent+log_height-math.log(delta)


@lru_cache(maxsize=4096)
def _cts1_tail(x,delta,lam,upper):
    # Gil-Pelaez inversion. Frequency scale follows either the SD or the
    # stable scale, so the fixed cutoff remains in the negligible CF tail.
    q=delta*lam
    unit=math.sqrt(delta/lam) if q>=1 else delta
    denominator=lam*unit

    def integrand(t):
        if t==0:
            return -x/unit
        w=_cts1_exponent(t/denominator,q)
        return math.exp(w.real)*math.sin(w.imag-t*x/unit)/t

    integral=quad(integrand,0.,40.,epsabs=2e-11,epsrel=2e-11,limit=500)[0]/np.pi
    result=.5+integral if upper else .5-integral
    if result < -1e-9 or result > 1+1e-9:
        raise ArithmeticError('index-one CTS CDF inversion left the probability range')
    return min(1.,max(0.,result))  # Remove only CDF cancellation roundoff.


class cts1_gen(rv_continuous):
    """Centered index-one CTS, S~cts1(delta,lam), with positive jumps and support R.

    The Levy measure is delta*exp(-lam*x)/x**2 dx for x>0; delta,lam>0.
    E[S]=0 and Var(S)=delta/lam. See cts1_cgf for Massing's logarithmic
    exponent. The class fixes the stability index at one; it has two shapes.
    PDF uses Esscher tilting and centered Fourier inversion, CDF uses CF
    inversion, and generic quantiles/sampling invert the CDF numerically.
    SciPy loc/scale act on S. This law is suitable as a log clock.

    All nonnegative integer moments exist and moment(n) computes them from
    the explicit cumulants via the complete Bell recurrence, without density
    quadrature. E[S**2]=delta/lam, E[S**3]=delta/lam**2, and
    E[S**4]=2*delta/lam**3+3*(delta/lam)**2. Exponential moments instead
    equal exp(cts1_cgf(s,delta,lam)), finite iff s<=lam. Fractional powers
    S**q need care because S can be negative; they are not the fractional
    absolute moments of the normal ratio documented in ntsr1.
    """

    def _argcheck(self,delta,lam):
        return (delta>0)&(lam>0)&np.isfinite(delta)&np.isfinite(lam)

    def _logpdf(self,x,delta,lam):
        return np.vectorize(_cts1_logpdf,otypes=[float])(x,delta,lam)

    def _pdf(self,x,delta,lam):
        return np.exp(self._logpdf(x,delta,lam))

    def _cdf(self,x,delta,lam):
        return np.vectorize(_cts1_tail,otypes=[float])(x,delta,lam,False)

    def _sf(self,x,delta,lam):
        return np.vectorize(_cts1_tail,otypes=[float])(x,delta,lam,True)

    def _munp(self,n,delta,lam):
        moments=[np.ones(np.broadcast(delta,lam).shape)]
        for order in range(1,int(n)+1):
            moment = np.zeros_like(moments[0])
            for j in range(1, order + 1):
                moment += comb(order - 1, j - 1) * cts1_cumulant(j, delta, lam) * moments[order - j]
            moments.append(moment)
        return moments[int(n)]

    def _stats(self,delta,lam):
        return 0.,delta/lam,1/np.sqrt(delta*lam),2/(delta*lam)


cts1=cts1_gen(name='cts1',shapes='delta, lam')


@lru_cache(maxsize=1024)
def _ntsr1_mixture(x,delta,lam,zeta,density):
    if x==0:
        return np.exp(cts1_cgf(-zeta,delta,lam))/np.sqrt(2*np.pi) if density else .5
    sd=math.sqrt(delta/lam)
    log_x=math.log(abs(x))

    def integrand(y):
        log_vol=zeta*sd*y
        log_arg=log_x-log_vol
        if log_arg>350:
            return 0.
        argument=math.exp(log_arg)
        conditional=norm.logpdf(argument)-log_vol if density else norm.logsf(argument)
        return np.exp(_cts1_logpdf(sd*y,delta,lam)+math.log(sd)+conditional)

    peak=zeta*log_x/sd
    bounds=sorted({-np.inf,-2.,0.,2.,peak,np.inf})
    return sum(quad(integrand,lo,hi,epsabs=1e-12,epsrel=3e-9,limit=200)[0]
               for lo,hi in zip(bounds[:-1],bounds[1:]))


class ntsr1_gen(rv_continuous):
    """Index-one normal ratio R=Z/exp(U), U=-zeta*S, S~cts1(delta,lam).

    Z is an independent standard normal and S is already centered.
    delta,lam>0 and zeta=+1 or -1. For the FCM/GSAS map, zeta=sign(k).
    The denominator has geometric mean one and R is symmetric.
    E[R**n]=(n-1)!!*exp(cts1_cgf(n*zeta,delta,lam)) for even n.
    For zeta=+1 absolute moments of order p exist iff p<=lam; for zeta=-1
    all positive orders exist. Odd moments vanish when they exist.
    PDF/CDF integrate the normal mixture; sampling uses numerical CTS1
    quantiles. SciPy loc/scale act on R. No finite positive-TSS c is used.

    For any real q>-1 with q*zeta<=lam, including the boundary,
    E[abs(R)**q] = 2**(q/2)*Gamma((q+1)/2)/sqrt(pi)
                  * exp(cts1_cgf(q*zeta,delta,lam)).
    For q<=-1 the absolute moment diverges because the normal factor does.
    With K(s)=cts1_cgf(s,delta,lam), variance=exp(K(2*zeta)) and
    excess kurtosis=3*expm1(K(4*zeta)-2*K(2*zeta)), when finite.
    For Y=loc+scale*R, E[abs(Y-loc)**q]=scale**q*E[abs(R)**q].
    The public SciPy moment(n) method evaluates integer signed raw moments;
    use the CGF formula above for fractional absolute moments. An odd moment
    is zero only if its absolute moment exists, not merely by symmetry.
    """

    def _argcheck(self,delta,lam,zeta):
        return cts1._argcheck(delta,lam)&(abs(zeta)==1)

    def _pdf(self,x,delta,lam,zeta):
        return np.vectorize(_ntsr1_mixture,otypes=[float])(x,delta,lam,zeta,True)

    def _cdf(self,x,delta,lam,zeta):
        tail=self._sf(abs(x),delta,lam,zeta)
        return np.where(x<=0,tail,1-tail)

    def _sf(self,x,delta,lam,zeta):
        tail=np.vectorize(_ntsr1_mixture,otypes=[float])(x,delta,lam,zeta,False)
        return np.where(x>=0,tail,1-tail)

    def _rvs(self,delta,lam,zeta,size=None,random_state=None):
        rng = self.random_state if random_state is None else random_state
        clock=cts1.rvs(delta,lam,size=size,random_state=rng)
        return rng.standard_normal(size)*np.exp(zeta*clock)

    def _munp(self,n,delta,lam,zeta):
        if n%2:
            return np.where((zeta<0)|(n<=lam),0.,np.nan)
        order=int(n)//2
        normal_moment=np.prod(np.arange(1,2*order,2,dtype=float))
        return normal_moment*np.exp(cts1_cgf(n*zeta,delta,lam))

    def _stats(self,delta,lam,zeta):
        log_m2=cts1_cgf(2*zeta,delta,lam)
        log_m4=cts1_cgf(4*zeta,delta,lam)
        mean=np.where((zeta<0)|(lam>=1),0.,np.nan)
        var=np.where(np.isnan(mean),np.nan,np.exp(log_m2))
        skew=np.where((zeta<0)|(lam>=3),0.,np.nan)
        with np.errstate(invalid='ignore'):
            kurt=np.where(np.isfinite(var),3*np.expm1(log_m4-2*log_m2),np.nan)
        return mean,var,skew,kurt


ntsr1=ntsr1_gen(name='ntsr1',shapes='delta, lam, zeta')


# Parameter maps: FCM -> positive TSS log clock, normalized GSAS -> NTSR.
def fcm2tss(alpha: float, k: float, eta=None, match: str = 'leading') -> tuple:
    """Map scalar unskewed FCM(alpha,k) to TSS shapes (a,c,lam) near LNN.

    IMPORTANT: the approximation is
        V / exp(E[log V]) ~= exp(-sign(k)*(T-E[T])),
    with T ~ tss(a,c,lam); it is not V ~= T. Book reference: fracdist.pdf,
    Section 7.6.3, Lemma 7.10, equations (7.42)-(7.47).

    eta is the distance of the TSS stability index a from one:
        eta = 1-a, with 0 < eta < 1 (e.g. eta=0.05 gives a=0.95).
    It controls the positive-TSS approximation in match='leading'. When
    eta=None, choose eta=h from the FCM inputs, retaining two free inputs.
    At fixed h, smaller eta moves the centered/reflected log clock toward
    the book's compensated index-one CTS law. Its raw positive-clock mean
    v*lam/eta diverges in that limit, so eta=0 is excluded here; use
    fcm2cts1/cts1 or gsas2ntsr1/ntsr1 for the exact index-one law. Changing eta
    preserves the leading log variance v and third cumulant -sign(k)*v*h;
    it changes the fourth and higher cumulants. For match='cumulants', a
    is determined by the exact log cumulants and eta must remain None.

    match='leading': h=1/z, z=k-1 for k>1 and z=|k| for k<0,
    v=(1/alpha-1/2)*h; set a=1-eta, lam=(1+eta)/h and
    c=v*lam**(2-a)/(a*(1-a)). Default eta=h needs h<1. This matches the
    book's leading log variance/third cumulant. eta -> 0 at fixed h gives
    the book's compensated index-one law; eta=h gives O(h**3) CGF error
    at fixed transform orders on the nondegenerate lognormal path.

    match='cumulants': match the exact FCM log cumulants k2,k3,k4.
    If r=k4*k2/k3**2 lies strictly in (3/2,2), the unique shapes are
    a=(2*r-3)/(r-1), lam=(2-a)*k2/abs(k3), c=k2*lam**(2-a)/(a*(1-a)).
    This is a moment match, not an identity of distributions. eta is unused
    and must be None. High precision protects the near-boundary ratio.

    Parameter count: the default leading map and the cumulant map each have
    only two free inputs, the FCM shapes (alpha,k). The returned (a,c,lam)
    are constrained: c is derived, rather than fitted independently, from
        c = v * lam**(2-a) / (a*(1-a)).
    Here v is the exact Var(log V)=k2 for the cumulant map, or the leading
    log-variance estimate above for the leading map. Thus c uses v as well
    as a and lam; it is not generally determined by a and lam alone.
    Supplying an explicit eta adds an approximation choice to the leading map.

    The returned raw TSS mean diverges toward LNN; center before exponentiating.
    """
    alpha, k = float(alpha), float(k)
    if not (np.isfinite(alpha) and 0 < alpha < 2 and np.isfinite(k) and (k > 1 or k < 0)):
        raise ValueError('requires 0 < alpha < 2 and k > 1 or k < 0')
    z = k-1 if k > 0 else -k
    h = 1/z
    if match == 'leading':
        eta = h if eta is None else float(eta)
        if not 0 < eta < 1:
            raise ValueError('eta must be in (0,1); default eta=h requires z>1')
        a = 1-eta
        v = (1/alpha-.5)*h
        lam = (1+eta)/h
        c = v*lam**(2-a)/(a*(1-a))
        return a, c, lam
    if match != 'cumulants' or eta is not None:
        raise ValueError("match must be 'leading' or 'cumulants'; cumulants requires eta=None")
    import mpmath as mp
    with mp.workdps(70):
        eps, zz = 1/mp.mpf(alpha), mp.mpf(z)
        values = {n: eps**n*mp.polygamma(n-1, eps*zz)
                  - mp.polygamma(n-1, zz/2)/mp.mpf(2)**n for n in (2,3,4)}
        ratio = values[4]*values[2]/values[3]**2
        if not mp.mpf('1.5') < ratio < 2:
            raise ValueError('FCM log cumulants do not admit this positive-TSS moment match')
        a = (2*ratio-3)/(ratio-1)
        # Recompute the intensity after rounding a and lam. Otherwise the
        # small rounding of a near one is amplified by 1/(1-a), perturbing
        # the variance and obscuring convergence of the returned float law.
        a_float = float(a)
        if not 0 < a_float < 1:
            raise ValueError('cumulant-match index rounds to a boundary; use the leading map')
        aa = mp.mpf(a_float)
        lam_float = float((2-aa)*values[2]/abs(values[3]))
        ll = mp.mpf(lam_float)
        c_float = float(values[2]*ll**(2-aa)/(aa*(1-aa)))
        return a_float, c_float, lam_float


def gsas2ntsr(alpha: float, k: float, eta=None, match: str = 'leading') -> tuple:
    """Return (a,c,lam,zeta) for ntsr approximating exp(m_F)*GSAS(alpha,k).

    m_F=E[log(frac_chi_mean(alpha,k))]. This geometric normalization is
    essential. To recover the unnormalized GSAS, use scale=exp(-m_F)
    when representable; keep m_F in log form near LNN to avoid overflow.
    ntsr(*gsas2ntsr(alpha,k)) has denominator exp(-zeta*(T-E[T])).
    """
    return (*fcm2tss(alpha, k, eta=eta, match=match), float(np.sign(k)))


def tss2fcm(a: float, c: float, lam: float, zeta: float = 1.0,
            match: str = 'leading') -> tuple:
    """Reverse the TSS log-clock map, returning FCM shapes (alpha,k).

    The positive clock loses the FCM branch: supply zeta=sign(k), either
    +1 (default) or -1. These maps concern exp(-zeta*(T-E[T])), not T.

    match='leading' is closed form. With v=Var(T), z=lam/(2-a), return
    alpha=1/(1/2+v*z), k=z+1 for zeta=+1 and k=-z for zeta=-1.
    This reverses fcm2tss(...,eta=1-a). The default forward eta=h is
    recovered only when 1-a=(2-a)/lam; arbitrary TSS shapes need that
    explicit eta to reproduce the original clock on the forward trip.

    match='cumulants' numerically matches the second and third log-clock
    cumulants at 70 digits, then checks the fourth to relative tolerance
    1e-10. It reverses the two-dimensional forward cumulant map; a general
    three-parameter TSS may lie outside its image and raises ValueError.
    No approximate projection or loc/scale conversion is performed.
    """
    a, c, lam, zeta = map(float, (a, c, lam, zeta))
    if not tss._argcheck(a, c, lam) or zeta not in (-1., 1.):
        raise ValueError('requires 0 < a < 1, finite c > 0, lam > 0, and zeta = +1 or -1')
    if match == 'leading':
        z = lam/(2-a)
        # Form v*z in log space so valid clock shapes cannot overflow it.
        log_delta = (np.log(c)+np.log(a)+np.log1p(-a)
                     +(a-2)*np.log(lam)+np.log(z))
        alpha = float(np.exp(-np.logaddexp(np.log(.5), log_delta)))
    elif match == 'cumulants':
        import mpmath as mp
        with mp.workdps(70):
            aa, cc, ll = map(mp.mpf, (a, c, lam))
            v = cc*aa*(1-aa)*ll**(aa-2)
            third = v*(2-aa)/ll
            fourth = third*(3-aa)/ll
            z0 = ll/(2-aa)

            def residual(log_delta, log_z):
                eps, zz = mp.mpf('.5')+mp.exp(log_delta), mp.exp(log_z)
                second_f = eps**2*mp.polygamma(1,eps*zz)-mp.polygamma(1,zz/2)/4
                third_f = -eps**3*mp.polygamma(2,eps*zz)+mp.polygamma(2,zz/2)/8
                return mp.log(second_f/v), mp.log(third_f/third)

            try:
                log_delta, log_z = mp.findroot(residual, (mp.log(v*z0),mp.log(z0)),
                                               tol=mp.mpf('1e-45'), maxsteps=50)
            except (ValueError, ZeroDivisionError, OverflowError) as exc:
                raise ValueError('FCM log-cumulant inversion did not converge') from exc
            eps, zz = mp.mpf('.5')+mp.exp(log_delta), mp.exp(log_z)
            fourth_f = eps**4*mp.polygamma(3,eps*zz)-mp.polygamma(3,zz/2)/16
            if abs(fourth_f/fourth-1) > mp.mpf('1e-10'):
                raise ValueError('TSS log cumulants lie outside the forward FCM cumulant map')
            alpha, z = float(1/eps), float(zz)
    else:
        raise ValueError("match must be 'leading' or 'cumulants'")
    k = z+1 if zeta > 0 else -z
    if not (0 < alpha < 2 and np.isfinite(k) and (k > 1 or k < 0)):
        raise ValueError('recovered FCM shapes are outside the representable domain')
    return alpha, k


def ntsr2gsas(a: float, c: float, lam: float, zeta: float,
              match: str = 'leading') -> tuple:
    """Reverse gsas2ntsr, returning GSAS shapes (alpha,k); see tss2fcm.

    This is a shape map for geometrically normalized GSAS. The corresponding
    raw GSAS is exp(-m_F)*NTSR, m_F=E[log FCM(alpha,k)]; loc/scale are not
    inferred or returned. The distributional correspondence is approximate
    even when the parameter round trip is exact.
    """
    return tss2fcm(a, c, lam, zeta=zeta, match=match)


def fcm2cts1(alpha: float, k: float) -> tuple:
    """Book's exact index-one log-law parameters (delta,lam) from FCM(alpha,k).

    delta=1/alpha-1/2 and lam=z, z=k-1 for k>1 or |k| for k<0.
    If S~cts1(*fcm2cts1(alpha,k)), log(V)-E[log V] ~= -sign(k)*S.
    The CTS law is evaluated exactly at index one; the approximation to FCM
    remains the book's leading O(h**3) CGF approximation, h=1/z.
    """
    alpha,k=float(alpha),float(k)
    if not (0<alpha<2 and np.isfinite(k) and (k>1 or k<0)):
        raise ValueError('requires 0 < alpha < 2 and k > 1 or k < 0')
    return 1/alpha-.5,k-1 if k>0 else -k


def gsas2ntsr1(alpha: float, k: float) -> tuple:
    """Return (delta,lam,zeta) for index-one NTSR approximating exp(m_F)*GSAS.

    m_F=E[log FCM(alpha,k)]. Restore raw GSAS scale with exp(-m_F),
    as in gsas2ntsr. This implements the book law directly, without eta.
    """
    return (*fcm2cts1(alpha,k),float(np.sign(k)))


def cts12fcm(delta: float, lam: float, zeta: float = 1.) -> tuple:
    """Reverse fcm2cts1: return (alpha,k); zeta restores the lost FCM branch."""
    delta,lam,zeta=map(float,(delta,lam,zeta))
    if not cts1._argcheck(delta,lam) or zeta not in (-1.,1.):
        raise ValueError('requires finite delta > 0, lam > 0, and zeta = +1 or -1')
    alpha,k=1/(delta+.5),lam+1 if zeta>0 else -lam
    if not (0<alpha<2 and (k>1 or k<0)):
        raise ValueError('recovered FCM shapes are outside the representable domain')
    return alpha,k


def ntsr12gsas(delta: float, lam: float, zeta: float) -> tuple:
    """Reverse gsas2ntsr1, returning GSAS shapes (alpha,k); loc/scale are separate."""
    return cts12fcm(delta,lam,zeta=zeta)
