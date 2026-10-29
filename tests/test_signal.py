import numpy as np
import pytest

from pdfcoherence.signal import (
    estimate_noise_rms,
    find_threshold_crossing,
    grid_step,
    rolling_rms,
    window_points,
)

r = np.arange(0.0, 200.0, 0.1)


def test_grid_step_rejects_nonuniform_grid():
    with pytest.raises(ValueError, match="not uniformly spaced"):
        grid_step(np.array([0.0, 0.1, 0.3, 0.4]))


def test_window_points_is_odd():
    assert window_points(r, 10.0) == 101
    assert window_points(r, 1.05) % 2 == 1


def test_rolling_rms_of_sine_including_edges():
    rms = rolling_rms(r, np.sin(2 * np.pi * r), 10.0)
    np.testing.assert_allclose(rms, 1 / np.sqrt(2), rtol=0.03)


def test_noise_estimate_of_white_noise():
    rng = np.random.default_rng(1)
    gr = 0.2 * rng.standard_normal(len(r))
    assert estimate_noise_rms(r, gr, 100.0, 200.0) == pytest.approx(
        0.2, rel=0.05
    )


def test_noise_estimate_warns_on_residual_signal():
    gr = np.exp(-r / 30.0) * np.sin(3 * r)
    with pytest.warns(UserWarning, match="still decreases"):
        estimate_noise_rms(r, gr, 100.0, 200.0)


def test_noise_estimate_needs_points():
    with pytest.raises(ValueError, match="fewer than 10 points"):
        estimate_noise_rms(r, r, 500.0, 600.0)


def test_threshold_crossing_requires_persistence():
    rms = np.where(r < 50.0, 1.0, 0.1)
    rms[(r > 30.0) & (r < 32.0)] = 0.1  # short dip, shorter than 15 A
    crossing = find_threshold_crossing(r, rms, 0.5, 10.0, 150.0, 15.0)
    assert crossing == pytest.approx(50.0, abs=0.05)


def test_threshold_crossing_at_end_of_scan():
    rms = np.where(r < 135.0, 1.0, 0.1)
    assert find_threshold_crossing(
        r, rms, 0.5, 10.0, 150.0, 15.0
    ) == pytest.approx(135.0, abs=0.05)


def test_threshold_crossing_none():
    assert np.isnan(
        find_threshold_crossing(r, np.ones_like(r), 0.5, 10.0, 150.0, 15.0)
    )
