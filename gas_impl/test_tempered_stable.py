"""Independent density, mixture and SciPy-interface checks for tempered clocks."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import kve
from scipy.stats import invgauss, norm

from .tempered_stable import ntsr, tss, tss_cf, tss_cumulant, tss_laplace


@pytest.mark.parametrize('alpha', [0.35, 0.75])
def test_clock_density_reproduces_its_laplace_transform_and_moments(alpha):
    """Fails when exponential tilting or stable scaling changes the clock's law."""
    c, lam = 1.2, 0.8
    clock = tss(alpha, c, lam)
    for s in [0.0, 0.5, 2.0]:
        actual = quad(lambda x: np.exp(-s*x) * clock.pdf(x), 0, np.inf,
                      epsabs=1e-10, epsrel=1e-8, limit=250)[0]
        expected = np.exp(-c * ((lam + s)**alpha - lam**alpha))
        assert actual == pytest.approx(expected, rel=2e-7)
    for order in [1, 2]:
        actual = quad(lambda x: x**order * clock.pdf(x), 0, np.inf,
                      epsabs=1e-10, epsrel=1e-8, limit=250)[0]
        assert actual == pytest.approx(clock.moment(order), rel=2e-7)
    x = clock.mean()
    assert clock.cdf(x) + clock.sf(x) == pytest.approx(1, abs=2e-8)
    assert clock.pdf(-1) == clock.cdf(-1) == 0
    assert np.isneginf(clock.logpdf(0))
    assert clock.sf(-1) == 1


@pytest.mark.parametrize('c,lam', [(0.4,2.),(1.2,.8),(3.,4.)])
def test_half_index_clock_matches_inverse_gaussian(c,lam):
    """Fails when the positive-clock Laplace coefficient or its moments are misparameterized."""
    clock=tss(.5,c,lam)
    ig=invgauss(mu=1/(c*np.sqrt(lam)),scale=c*c/2)
    x=ig.ppf([1e-4,.1,.5,.9,.9999])
    for method in ['pdf','logpdf','cdf','sf']:
        np.testing.assert_allclose(getattr(clock,method)(x),getattr(ig,method)(x),rtol=2e-12)
    np.testing.assert_allclose(clock.stats(moments='mvsk'),ig.stats(moments='mvsk'),rtol=2e-12)
    for order in [0,1,2,4,6]:
        mean,z=c/(2*np.sqrt(lam)),c*np.sqrt(lam)
        expected=mean**order*kve(order-.5,z)/kve(.5,z)
        assert clock.moment(order)==pytest.approx(expected,rel=2e-11)


@pytest.mark.parametrize('zeta', [-1., 1.])
def test_exponential_clock_mixture_matches_independent_ig_integral(zeta):
    """Fails when NTSR uses T directly, misses centering/reflection, or the PDF Jacobian."""
    c, lam = 1.2, 5.
    dist = ntsr(.5, c, lam, zeta)
    mean, shape = c/(2*np.sqrt(lam)), c*c/2
    for x in [0., .7, 4.]:
        def integrand(t):
            log_ig = (.5*np.log(shape/(2*np.pi))-1.5*np.log(t)
                      -shape*(t-mean)**2/(2*mean*mean*t))
            log_vol = zeta*(t-mean)
            log_arg = np.log(x)-log_vol if x else -np.inf
            if log_arg > 350:
                return 0.
            return np.exp(log_ig-log_vol+norm.logpdf(np.exp(log_arg)))
        expected = quad(integrand, 0, np.inf, epsabs=1e-12, epsrel=1e-10)[0]
        assert dist.pdf(x) == pytest.approx(expected, rel=3e-8)
    K = lambda s: c*(np.sqrt(lam)-np.sqrt(lam-s))-s*mean
    m2, m4 = np.exp(K(2*zeta)), 3*np.exp(K(4*zeta))
    assert dist.var() == pytest.approx(m2, rel=1e-11)
    assert dist.moment(4) == pytest.approx(m4, rel=1e-11)
    assert dist.stats(moments='k') == pytest.approx(m4/m2**2-3, rel=1e-11)
    assert dist.mean() == dist.stats(moments='s') == 0
    np.testing.assert_allclose(dist.cdf([-.3,-2]),dist.sf([.3,2]),rtol=1e-12)


@pytest.mark.parametrize('alpha,zeta', [(.35,1.),(.65,-1.),(.9,1.)])
def test_exponential_clock_cdf_is_continuous_at_zero(alpha,zeta):
    """Fails when near-zero return arguments skip clock probability mass."""
    c,lam=1.2,5.
    dist=ntsr(alpha,c,lam,zeta)
    mean=c*alpha*lam**(alpha-1)
    log_peak=-c*((lam+zeta)**alpha-lam**alpha)+zeta*mean
    peak=np.exp(log_peak)/np.sqrt(2*np.pi)
    for h in [1e-12,1e-4]:
        left,right=dist.cdf([-h,h])
        assert abs(right-.5)<=h*peak*1.001+1e-12
        assert left+right==pytest.approx(1.,abs=1e-13)
        if h==1e-4:
            assert (right-left)/(2*h)==pytest.approx(peak,rel=3e-6)


@pytest.mark.parametrize('distribution', [tss, ntsr])
def test_scipy_frozen_broadcast_quantiles_and_location_scale(distribution):
    """Fails when frozen/broadcast distributions or non-default loc/scale change the law."""
    alpha, c, lam = .65, 1.2, 5. if distribution is ntsr else .8
    extra = {'zeta':1.} if distribution is ntsr else {}
    dist = distribution(alpha, c, lam, **extra)
    q = np.array([.1, .5, .9])
    x = dist.ppf(q)
    np.testing.assert_allclose(dist.cdf(x), q, atol=3e-8)
    shifted = distribution(alpha, c, lam, loc=2., scale=3., **extra)
    np.testing.assert_allclose(shifted.pdf(2+3*x), dist.pdf(x)/3, rtol=1e-10)
    np.testing.assert_allclose(shifted.cdf(2+3*x), dist.cdf(x), rtol=1e-10)
    broadcast = distribution.pdf(x[:2], alpha, [c, c*2], lam, **extra)
    np.testing.assert_allclose(broadcast, [distribution.pdf(x[0], alpha, c, lam, **extra),
                                         distribution.pdf(x[1], alpha, c*2, lam, **extra)])
    assert np.isnan(distribution.pdf(1, 1., c, lam, **extra))
    assert np.isnan(distribution.pdf(1, alpha, np.inf, lam, **extra))
    assert dist.rvs(size=(0, 2), random_state=1).shape == (0, 2)


@pytest.mark.parametrize('alpha,c,lam', [(.35, 1.2, .8), (.7, 3., 3.)])
def test_samplers_follow_laplace_and_normal_mixture_laws(alpha, c, lam):
    """Fails when rejection tilts the wrong way, splitting changes intensity, or mixtures share normals."""
    seed, count = 1234, 40_000
    clocks = tss.rvs(alpha, c, lam, size=count, random_state=seed)
    assert (clocks > 0).all()
    for s in [.5, 2.]:
        observed = np.exp(-s*clocks)
        assert abs(observed.mean()-tss_laplace(s, alpha, c, lam)) < 5*observed.std()/np.sqrt(count)
    # Independent reference CDF at a few fixed points, avoiding an expensive KS
    # evaluation of the quadrature CDF at every simulated observation.
    inverse_returns = ntsr.rvs(alpha, c, lam, 1., size=count, random_state=seed)
    for q in [.1, .9]:
        threshold = ntsr.ppf(q, alpha, c, lam, 1.)
        actual = np.mean(inverse_returns <= threshold)
        assert abs(actual-q) < 5*np.sqrt(q*(1-q)/count)
    for distribution in [tss, ntsr]:
        extra = {'zeta':1.} if distribution is ntsr else {}
        first = distribution.rvs(alpha, c, lam, size=(3, 2), random_state=seed, **extra)
        second = distribution.rvs(alpha, c, lam, size=(3, 2), random_state=seed, **extra)
        np.testing.assert_array_equal(first, second)
        assert np.isfinite(distribution.rvs(alpha, c, lam, random_state=seed, **extra))
        assert distribution.rvs([alpha, .5], c, lam, size=(3, 2), random_state=seed, **extra).shape == (3, 2)


def test_clock_cumulants_and_log_clock_mixture_stats_broadcast():
    """Fails when public moment helpers or log-clock-mixture statistics lose shape broadcasting."""
    a = np.array([.35, .7])
    c, lam = 1.2, .8
    np.testing.assert_allclose(tss_cumulant(1, a, c, lam), c*a*lam**(a-1))
    np.testing.assert_allclose(tss_cumulant(2, a, c, lam), c*a*(1-a)*lam**(a-2))
    for distribution in [tss, ntsr]:
        extra = {'zeta':-1.} if distribution is ntsr else {}
        for actual, single in zip(distribution.stats(a, c, lam, moments='mvsk', **extra),
                                  np.array([distribution.stats(ai, c, lam, moments='mvsk', **extra) for ai in a]).T):
            np.testing.assert_allclose(actual, single, rtol=1e-12)
    for n in [0, -1, 1.5]:
        with pytest.raises(ValueError):
            tss_cumulant(n, .5, c, lam)


@pytest.mark.parametrize('alpha', [.3, .75, .95])
def test_concentrated_clock_density_and_cdf_match_fourier_inversion(alpha):
    """Fails when a small-x stable density error is exponentially amplified by strong tempering."""
    lam = 50.
    c = lam**(1-alpha)/alpha  # E[T]=1, Var(T)=(1-alpha)/lam.
    clock = tss(alpha, c, lam)
    mean, sd = 1., np.sqrt((1-alpha)/lam)
    for z in [-1., 0., 1.]:
        x = mean + z*sd
        # Independent CF inversion in standardized frequency, with no stable
        # density or Kanter angle integral on the reference side.
        cf = lambda v: tss_cf(v/sd, alpha, c, lam)*np.exp(-1j*v*x/sd)
        pdf_ref = quad(lambda v: cf(v).real, 0, np.inf, epsabs=1e-10, limit=250)[0]/(np.pi*sd)
        cdf_ref = .5-quad(lambda v: cf(v).imag/v if v != 0 else (mean-x)/sd,
                          0, np.inf, epsabs=1e-10, limit=250)[0]/np.pi
        assert clock.pdf(x) == pytest.approx(pdf_ref, rel=2e-8)
        assert clock.cdf(x) == pytest.approx(cdf_ref, abs=2e-8)
