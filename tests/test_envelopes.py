import numpy as np
import pytest

from pdfcoherence.envelopes import (
    exponential,
    get_envelope,
    lognormal_sphere,
    qdamp_envelope,
    sphere,
)

r = np.linspace(0, 300, 3001)


def test_sphere_limits_and_monotonic():
    f = sphere(r, 100.0)
    assert f[0] == pytest.approx(1.0)
    assert np.all(f[r >= 100.0] == 0.0)
    assert np.all(np.diff(f) <= 0)


def test_exponential_at_xi():
    assert exponential(np.array([25.0]), 25.0)[0] == pytest.approx(np.exp(-1))


def test_qdamp_envelope_zero_is_one():
    assert np.all(qdamp_envelope(r, 0.0) == 1.0)


def test_lognormal_reduces_to_sphere():
    np.testing.assert_allclose(lognormal_sphere(r, 80.0, 0.0), sphere(r, 80.0))
    np.testing.assert_allclose(
        lognormal_sphere(r, 80.0, 1e-4), sphere(r, 80.0), atol=1e-3
    )


@pytest.mark.parametrize("polydispersity", [0.1, 0.3, 0.6])
def test_lognormal_matches_numerical_average(polydispersity):
    # <D**3 sphere(r, D)> / <D**3> over a lognormal number distribution
    mean = 100.0
    s2 = np.log1p(polydispersity**2)
    d = np.geomspace(0.5, 2e4, 200001)
    number = np.exp(-((np.log(d) - np.log(mean) + s2 / 2) ** 2) / (2 * s2)) / d
    weight = number * d**3
    expected = np.array(
        [np.trapezoid(weight * sphere(x, d), d) for x in r[::50]]
    ) / np.trapezoid(weight, d)
    np.testing.assert_allclose(
        lognormal_sphere(r[::50], mean, polydispersity), expected, atol=1e-6
    )


def test_lognormal_has_longer_tail():
    narrow = lognormal_sphere(r, 100.0, 0.1)
    broad = lognormal_sphere(r, 100.0, 0.4)
    i150 = np.argmin(abs(r - 150.0))
    assert broad[0] == pytest.approx(1.0)
    assert broad[i150] > narrow[i150]


def test_get_envelope_unknown_shape():
    with pytest.raises(ValueError, match="Unknown shape"):
        get_envelope("cube")
