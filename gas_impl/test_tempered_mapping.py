"""Parameter-map and centered-log-clock regressions, independent of plotting."""

import mpmath as mp
import numpy as np
import pytest

from .tempered_stable import (
    fcm2tss, gsas2ntsr, ntsr, ntsr2gsas, tss2fcm, tss_centered_cgf, tss_cumulant,
)


@pytest.mark.parametrize('zeta', [-1., 1.])
@pytest.mark.parametrize('h', [.05, .0125])
def test_cumulant_map_reproduces_exact_fcm_log_shape(zeta,h):
    """Fails for a wrong branch/index, or when rounding a near one perturbs clock intensity."""
    for v in [.04, .25, 1.]:
        k = 1+1/h if zeta > 0 else -1/h
        alpha = 1/(.5+v/h)
        a, c, lam, branch = gsas2ntsr(alpha, k, match='cumulants')
        assert 0 < a < 1 and branch == zeta
        with mp.workdps(70):
            eps, z = 1/mp.mpf(alpha), mp.mpf(1/h)
            for n in [2, 3, 4]:
                exact = zeta**n*(eps**n*mp.polygamma(n-1, eps*z)
                                 -mp.polygamma(n-1,z/2)/mp.mpf(2)**n)
                actual = (-zeta)**n*tss_cumulant(n,a,c,lam)
                assert actual == pytest.approx(float(exact),rel=5e-13)


def test_leading_map_recovers_book_log_law_as_index_approaches_one():
    """Fails when centering, reflection or the divergent TSS intensity changes the index-one limit."""
    v, h = .25, .1
    alpha, k = 1/(.5+v/h), 1+1/h
    s = np.array([-.7, 1.3, 2j])
    q = h*s
    reference = v/h**2*((1+q)*np.log1p(q)-q)
    errors = []
    for eta in [.1,.01,.001]:
        a,c,lam = fcm2tss(alpha,k,eta=eta)
        assert tss_cumulant(2,a,c,lam) == pytest.approx(v,rel=1e-12)
        assert -tss_cumulant(3,a,c,lam) == pytest.approx(-v*h,rel=1e-12)
        errors.append(np.max(np.abs(tss_centered_cgf(-s,a,c,lam)-reference)))
    assert errors[2] < errors[1]/8 < errors[0]/64


def test_log_clock_moment_domain_is_respected():
    """Fails when exponential-clock moments are falsely reported finite beyond their MGF boundary."""
    positive_branch = ntsr(.5,1.2,3.,1.)
    assert np.isfinite(positive_branch.var())
    assert np.isinf(positive_branch.moment(4))
    assert np.isinf(positive_branch.stats(moments='k'))
    assert np.isnan(positive_branch.moment(5))
    assert np.isfinite(positive_branch.moment(3))  # boundary order is finite
    negative_branch = ntsr(.5,1.2,.8,-1.)
    assert np.isfinite(negative_branch.moment(8))
    assert np.isinf(negative_branch.pdf(0))  # infinite peak is allowed


def test_centered_transform_retains_precision_near_index_one():
    """Fourier tails must survive cancellation of the divergent clock drift."""
    shapes = fcm2tss(1/(.5+.25/.025), 41., match='cumulants')
    orders = np.array([0., .01j, 1j, 10j, 50j, 80j, shapes[2]], dtype=complex)
    with mp.workdps(70):
        a,c,lam = map(mp.mpf,shapes)
        expected = [complex(-c*((lam-complex(s))**a-lam**a
                               +a*lam**(a-1)*complex(s))) for s in orders]
    np.testing.assert_allclose(tss_centered_cgf(orders,*shapes),expected,
                               rtol=3e-13,atol=1e-14)


@pytest.mark.parametrize('alpha,k,kwargs', [(2.,11.,{}),(.5,1.,{}),(.5,0.,{}),
                                         (.5,11.,{'eta':1.}),
                                         (.5,11.,{'match':'unknown'})])
def test_invalid_mapping_does_not_silently_choose_another_branch(alpha,k,kwargs):
    """Fails when invalid inputs are silently converted into another clock or branch."""
    with pytest.raises(ValueError):
        fcm2tss(alpha,k,**kwargs)


@pytest.mark.parametrize('match', ['leading','cumulants'])
@pytest.mark.parametrize('zeta', [-1.,1.])
def test_reverse_maps_recover_fcm_and_gsas_shapes(match,zeta):
    """Fails when a branch, geometric normalization, or near-LNN conditioning breaks a parameter round trip."""
    for alpha,z in [(.26289616784422354,13.575202096192339),(.7,7.),
                    (1.99,19.),(1/(.5+.25/.0125),80.)]:
        k=z+1 if zeta > 0 else -z
        shapes=fcm2tss(alpha,k,match=match)
        np.testing.assert_allclose(tss2fcm(*shapes,zeta=zeta,match=match),
                                   (alpha,k),rtol=2e-11)
        np.testing.assert_allclose(ntsr2gsas(*gsas2ntsr(alpha,k,match=match),match=match),
                                   (alpha,k),rtol=2e-11)


def test_reverse_leading_map_preserves_custom_eta_and_missing_branch():
    """Fails when an arbitrary clock is treated as the eta=h surface or its lost branch is silently inferred."""
    # Var(T)=1/4 and z=6/(2-1/2)=4, so alpha_F=1/(1/2+1)=2/3.
    shapes=(.5,6**1.5,6.)
    np.testing.assert_allclose(tss2fcm(*shapes),(2/3,5.))
    np.testing.assert_allclose(tss2fcm(*shapes,zeta=-1),(2/3,-4.))
    for zeta in [-1.,1.]:
        alpha,k=tss2fcm(*shapes,zeta=zeta)
        np.testing.assert_allclose(fcm2tss(alpha,k,eta=1-shapes[0]),shapes,rtol=2e-14)
    assert not np.allclose(fcm2tss(*tss2fcm(*shapes)),shapes)


def test_reverse_cumulant_map_rejects_a_clock_off_the_fcm_surface():
    """Fails when matching two cumulants is silently presented as an inverse of the three-cumulant map."""
    with pytest.raises(ValueError,match='outside the forward FCM cumulant map'):
        tss2fcm(.5,1.,2.,match='cumulants')


@pytest.mark.parametrize('shapes,kwargs', [((0.,1.,2.),{}),((1.,1.,2.),{}),
    ((.5,0.,2.),{}),((.5,np.nan,2.),{}),((.5,1.,np.inf),{}),
    ((.5,1.,2.),{'zeta':0}),((.5,1.,2.),{'match':'unknown'}),
    ((.5,1e300,1e-300),{})])
def test_reverse_maps_reject_invalid_or_unrepresentable_shapes(shapes,kwargs):
    """Fails when an invalid clock or an unrepresentable FCM is returned as valid shapes."""
    with pytest.raises(ValueError):
        tss2fcm(*shapes,**kwargs)
