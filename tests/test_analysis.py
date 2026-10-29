import numpy as np
import pytest
from conftest import make_pdf

from pdfcoherence.analysis import analyze


def test_sphere_fit_recovers_diameter(synthetic_pdf):
    result = analyze(*synthetic_pdf, qdamp=0.0, shape="sphere")
    assert result.status == "Success"
    assert result.fit.param == pytest.approx(60.0, rel=0.1)
    assert result.noise_rms == pytest.approx(0.05, rel=0.1)
    assert result.r_model < result.fit.param
    assert result.r_model == pytest.approx(result.r_thresh, abs=10.0)


def test_sphere_fit_with_qdamp():
    r, gr = make_pdf(diameter=80.0, qdamp=0.02, seed=3)
    result = analyze(r, gr, qdamp=0.02, shape="sphere")
    assert result.fit.param == pytest.approx(80.0, rel=0.15)


def test_lognormal_zero_width_equals_sphere(synthetic_pdf):
    sph = analyze(*synthetic_pdf, qdamp=0.0, shape="sphere")
    logn = analyze(*synthetic_pdf, qdamp=0.0, shape="lognormal")
    assert logn.fit.param == pytest.approx(sph.fit.param, rel=1e-6)


def test_r_model_depends_weakly_on_shape(synthetic_pdf):
    r_models = [
        analyze(*synthetic_pdf, 0.0, shape=s, polydispersity=0.3).r_model
        for s in ("sphere", "lognormal", "exponential")
    ]
    assert max(r_models) / min(r_models) < 1.25


def test_shape_none_reports_only_threshold(synthetic_pdf):
    result = analyze(*synthetic_pdf, qdamp=0.0, shape="none")
    assert result.fit is None and np.isnan(result.r_model)
    assert 30.0 < result.r_thresh < 60.0


def test_noise_free_data_raise():
    r, gr = make_pdf(noise=0.0)
    with pytest.raises(ValueError, match="noise_level"):
        analyze(r, gr, qdamp=0.0)


def test_noise_level_override():
    r, gr = make_pdf(noise=0.0)
    result = analyze(r, gr, qdamp=0.0, noise_level=0.05)
    assert result.noise_rms == 0.05
    assert result.fit.param == pytest.approx(60.0, rel=0.1)


def test_unconstrained_size_is_flagged():
    r, gr = make_pdf(diameter=5000.0, qdamp=0.02, seed=4)
    result = analyze(r, gr, qdamp=0.02, shape="sphere")
    assert result.fit.unconstrained
    assert "do not constrain" in result.status
