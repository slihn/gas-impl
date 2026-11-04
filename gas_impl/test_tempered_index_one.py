"""Independent index-one CTS/log-clock checks, including the book limit."""

import mpmath as mp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from .tempered_stable import (
    cts1, cts1_cgf, cts1_cf, cts1_cumulant, cts12fcm, fcm2cts1,
    fcm2tss, gsas2ntsr1, ntsr1, ntsr12gsas, tss, tss_centered_cgf,
)


@pytest.mark.parametrize('zeta', [-1.,1.])
def test_index_one_map_is_the_book_log_law_and_has_a_reverse(zeta):
    """Fails when index one is replaced by a small eta, loses reflection, or changes book normalization."""
    for alpha,z in [(.26289616784422354,13.575202096192339),(.4,2.),(.05,80.)]:
        k=z+1 if zeta>0 else -z
        delta,lam=fcm2cts1(alpha,k)
        assert delta==pytest.approx(1/alpha-.5) and lam==z
        np.testing.assert_allclose(cts12fcm(delta,lam,zeta),(alpha,k),rtol=2e-14)
        np.testing.assert_allclose(ntsr12gsas(*gsas2ntsr1(alpha,k)),(alpha,k),rtol=2e-14)
        for s in [-.5,1.,2j]:
            with mp.workdps(70):
                dd,ll,ss=mp.mpf(delta),mp.mpf(lam),mp.mpc(s)
                q=mp.mpf(zeta)*ss/ll
                reference=complex(dd*ll*((1+q)*mp.log1p(q)-q))
            assert cts1_cgf(-zeta*s,delta,lam)==pytest.approx(reference,rel=2e-13,abs=1e-14)
    assert cts1_cgf(lam,delta,lam)==pytest.approx(delta*lam)
    assert cts1_cgf(complex(lam),delta,lam)==pytest.approx(delta*lam)
    assert np.isinf(cts1_cgf(lam+.1,delta,lam))


@pytest.mark.parametrize('delta,lam', [(.05,.2),(1.,1.),(2.5,10.),(50.,200.)])
def test_index_one_density_and_cdf_match_independent_cf_inversion(delta,lam):
    """Fails when stable alpha-one scale shifts or a strong Esscher tilt corrupt the clock density."""
    sd=np.sqrt(delta/lam)
    unit=sd if delta*lam>=1 else delta
    for z in [-1.,0.,2.]:
        x=z*sd

        def phi(t):
            omega=t/unit
            if t==0:
                return 1.+0j
            exponent=delta*((lam-1j*omega)*np.log(1-1j*omega/lam)+1j*omega)
            return np.exp(exponent-1j*omega*x)

        pdf=quad(lambda t:phi(t).real,0,40,epsabs=2e-10,limit=500)[0]/(np.pi*unit)
        cdf=.5-quad(lambda t:phi(t).imag/t if t else -x/unit,
                    0,40,epsabs=2e-10,limit=500)[0]/np.pi
        assert cts1.pdf(x,delta,lam)==pytest.approx(pdf,rel=2e-7,abs=2e-9)
        assert cts1.cdf(x,delta,lam)==pytest.approx(cdf,abs=2e-9)
        assert cts1.cdf(x,delta,lam)+cts1.sf(x,delta,lam)==pytest.approx(1.,abs=1e-12)


def test_index_one_positive_density_has_unit_mass_and_the_claimed_moments():
    """Fails when a correct-looking CF is paired with an unnormalized or miscentered numerical PDF."""
    delta,lam=2.5,10.
    dist=cts1(delta,lam)
    for n in [0,1,2,3,4]:
        observed=quad(lambda x:x**n*dist.pdf(x),-np.inf,np.inf,
                      epsabs=1e-9,limit=200)[0]
        assert observed==pytest.approx(dist.moment(n),rel=2e-8,abs=1e-9)
    for s in [-lam/2,lam/2]:
        observed=quad(lambda x:np.exp(s*x+dist.logpdf(x)),-np.inf,np.inf,
                      epsabs=1e-9,limit=200)[0]
        assert observed==pytest.approx(np.exp(cts1_cgf(s,delta,lam)),rel=2e-8)
    np.testing.assert_allclose(dist.stats(moments='mvsk'),
                               [0.,delta/lam,1/np.sqrt(delta*lam),2/(delta*lam)])
    for n in [2,3,4]:
        assert cts1_cumulant(n,delta,lam)==pytest.approx(
            [delta/lam,delta/lam**2,2*delta/lam**3][n-2])


def test_positive_tss_bridge_converges_to_the_exact_index_one_clock():
    """Fails when the finite index-one law disagrees with the centered positive-TSS limit."""
    alpha,k=.4,11.
    shapes=fcm2cts1(alpha,k)
    orders=np.array([-.5,1.,2j])
    errors=[]
    for eta in [.1,.01,.001]:
        bridge=fcm2tss(alpha,k,eta=eta)
        errors.append(np.max(abs(tss_centered_cgf(orders,*bridge)-cts1_cgf(orders,*shapes))))
    assert errors[2]<errors[1]/8<errors[0]/64
    assert np.isnan(tss.pdf(1.,1.,1.,1.))  # Index one is a centered CTS, not a positive TSS.


@pytest.mark.parametrize('zeta', [-1.,1.])
def test_index_one_normal_ratio_matches_an_independent_fourier_clock_grid(zeta):
    """Fails when the exponential denominator, branch sign, or normal-mixture Jacobian is wrong."""
    delta,lam=2.5,10.
    n,width=4096,24.
    u=np.arange(-n//2,n//2)*(width/n)
    omega=2*np.pi*np.fft.fftfreq(n,d=width/n)
    p=np.exp(delta*((lam-1j*omega)*np.log(1-1j*omega/lam)+1j*omega))
    fu=np.fft.fftshift(np.fft.fft(p).real)/width
    inv_vol=np.exp(-zeta*u)
    dist=ntsr1(delta,lam,zeta)
    for x in [.0,.7,3.]:
        expected_pdf=np.sum(norm.pdf(x*inv_vol)*inv_vol*fu)*width/n
        expected_cdf=np.sum(norm.cdf(x*inv_vol)*fu)*width/n
        assert dist.pdf(x)==pytest.approx(expected_pdf,rel=2e-8,abs=1e-9)
        assert dist.cdf(x)==pytest.approx(expected_cdf,abs=1e-9)
    for order in [2,4]:
        normal_moment=1 if order==2 else 3
        expected=normal_moment*np.exp(cts1_cgf(order*zeta,delta,lam))
        assert dist.moment(order)==pytest.approx(expected,rel=1e-12)
    np.testing.assert_allclose(dist.cdf([-1.,-.1]),dist.sf([1.,.1]))


@pytest.mark.parametrize('distribution,shapes', [(cts1,(2.5,10.)),(ntsr1,(2.5,10.,1.))])
def test_index_one_scipy_quantiles_broadcasting_and_location_scale(distribution,shapes):
    """Fails when frozen index-one laws lose quantile inversion, shape broadcasting, or SciPy scale semantics."""
    dist=distribution(*shapes)
    x=dist.ppf([.1,.9])
    np.testing.assert_allclose(dist.cdf(x),[.1,.9],atol=2e-8)
    shifted=distribution(*shapes,loc=2.,scale=3.)
    np.testing.assert_allclose(shifted.pdf(2+3*x),dist.pdf(x)/3,rtol=2e-8)
    np.testing.assert_allclose(shifted.cdf(2+3*x),dist.cdf(x),atol=2e-9)
    extra=shapes[2:]
    broadcast=distribution.pdf(x,[2.5,3.],10.,*extra)
    np.testing.assert_allclose(broadcast,[distribution.pdf(x[0],2.5,10.,*extra),
                                         distribution.pdf(x[1],3.,10.,*extra)])
    assert dist.rvs(size=(0,2),random_state=1).shape==(0,2)
    np.testing.assert_array_equal(dist.rvs(size=(2,2),random_state=1),dist.rvs(size=(2,2),random_state=1))


def test_index_one_inverse_cdf_sampler_matches_the_clock_cf():
    """Fails when numerical quantiles sample the wrong alpha-one stable scale, drift, or tempering."""
    values=cts1.rvs(2.5,10.,size=1024,random_state=1234)
    for t in [.5,1.5]:
        observed=np.exp(1j*t*values)
        expected=cts1_cf(t,2.5,10.)
        assert abs(observed.mean()-expected)<5*np.std(observed)/np.sqrt(len(values))


def test_index_one_moment_domains_and_invalid_shapes():
    """Fails when the exact boundary is rejected or nonexistent exponential-clock moments are reported finite."""
    positive=ntsr1(1.,3.,1.)
    assert np.isfinite(positive.moment(3)) and np.isinf(positive.moment(4))
    assert np.isnan(positive.moment(5))
    negative=ntsr1(1.,.8,-1.)
    assert np.isfinite(negative.moment(8)) and np.isinf(negative.pdf(0))
    for delta,lam in [(0.,1.),(1.,0.),(np.inf,1.),(1.,np.nan)]:
        assert np.isnan(cts1.pdf(0.,delta,lam))
        with pytest.raises(ValueError): cts12fcm(delta,lam)
    assert np.isnan(ntsr1.pdf(0.,1.,1.,0.))
    for alpha,k in [(2.,10.),(.5,0.),(.5,1.)]:
        with pytest.raises(ValueError): fcm2cts1(alpha,k)
    with pytest.raises(ValueError): cts1_cumulant(0,1.,1.)
