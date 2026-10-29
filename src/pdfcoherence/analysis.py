"""Coherence-length analysis of a single PDF."""

from dataclasses import dataclass, field

import numpy as np

from pdfcoherence.envelopes import SHAPES, get_envelope, qdamp_envelope
from pdfcoherence.fitting import EnvelopeFit, fit_envelope, model_rms
from pdfcoherence.signal import (
    default_noise_range,
    estimate_noise_rms,
    find_threshold_crossing,
    grid_step,
    rolling_rms,
    window_points,
)


@dataclass
class CoherenceResult:
    """Outcome of the coherence-length analysis of one PDF.

    Attributes
    ----------
    r, gr, rms : numpy.ndarray
        The r-grid in Angstrom, G(r), and its rolling RMS.
    noise_rms : float
        The noise floor in the units of G(r).
    threshold : float
        The detection threshold ``threshold_factor * noise_rms``.
    r_thresh : float
        The distance in Angstrom where the measured rolling RMS falls
        below the threshold and stays there (NaN if it never does).
    r_model : float
        The distance in Angstrom where the fitted RMS model falls below
        the threshold (NaN without a successful fit).
    shape : str
        The envelope model.
    fit : EnvelopeFit or None
        The refined envelope parameters.
    model_curve, envelope_curve : numpy.ndarray or None
        The fitted RMS model and the signal envelope ``A f(r) B(r)``.
    settings : dict
        The analysis settings, kept for plotting and reporting.
    status : str
        "Success" or a description of what went wrong.
    """

    r: np.ndarray
    gr: np.ndarray
    rms: np.ndarray
    noise_rms: float
    threshold: float
    r_thresh: float
    shape: str
    r_model: float = np.nan
    fit: EnvelopeFit | None = None
    model_curve: np.ndarray | None = None
    envelope_curve: np.ndarray | None = None
    settings: dict = field(default_factory=dict)
    status: str = "Success"


def analyze(
    r,
    gr,
    qdamp,
    shape="sphere",
    polydispersity=0.0,
    fit_min=15.0,
    fit_max=None,
    noise_range=None,
    noise_level=None,
    window=10.0,
    threshold_factor=1.15,
    persistence=15.0,
):
    """Estimate the coherence length of a material from its PDF.

    Parameters
    ----------
    r : numpy.ndarray
        The uniformly spaced real-space distances in Angstrom.
    gr : numpy.ndarray
        The PDF values G(r).
    qdamp : float
        The instrumental damping parameter in inverse Angstrom, refined
        on a calibrant measured with the same setup.
    shape : str, optional
        The envelope model: "none", "sphere", "lognormal" or
        "exponential".
    polydispersity : float, optional
        The relative width of the diameter distribution for the
        "lognormal" shape.
    fit_min : float, optional
        The lower limit of the fit and threshold scan in Angstrom. Set it
        beyond intra-cluster or intra-unit-cell distances.
    fit_max : float, optional
        The upper limit of the fit in Angstrom. By default
        ``1.4 * r_thresh`` (at least six windows above ``fit_min``),
        capped at the start of the noise range.
    noise_range : tuple of float, optional
        The r-range in Angstrom used to estimate the noise floor. By
        default the last 20 % of the data.
    noise_level : float, optional
        The RMS of white noise, in the units of G(r), added in quadrature
        to the rolling RMS and used as the noise floor instead of the
        estimate from ``noise_range``. Required for noise-free
        (simulated) data.
    window : float, optional
        The width of the rolling RMS window in Angstrom.
    threshold_factor : float, optional
        The detection threshold as a multiple of the noise floor.
    persistence : float, optional
        The distance in Angstrom the measured RMS must stay below the
        threshold to define ``r_thresh``.

    Returns
    -------
    CoherenceResult
        The threshold and model distances, fit and curves.
    """
    dr = grid_step(r)
    if noise_range is None:
        noise_range = default_noise_range(r)
    rms = rolling_rms(r, gr, window)
    if noise_level is None:
        noise_rms = estimate_noise_rms(r, gr, *noise_range)
        signal_max = rms[r >= fit_min].max()
        if noise_rms < 1e-3 * signal_max:
            raise ValueError(
                f"The noise floor ({noise_rms:.2e}) is effectively zero "
                f"compared to the signal ({signal_max:.2e}), as for "
                "noise-free simulated data. The coherence length is then "
                "set by numerical precision. Pass noise_level "
                "(--noise-level) to impose a realistic noise floor."
            )
    else:
        noise_rms = float(noise_level)
        rms = np.sqrt(rms**2 + noise_rms**2)
    threshold = threshold_factor * noise_rms
    r_thresh = find_threshold_crossing(
        r, rms, threshold, fit_min, noise_range[0], persistence
    )
    result = CoherenceResult(
        r=r,
        gr=gr,
        rms=rms,
        noise_rms=noise_rms,
        threshold=threshold,
        r_thresh=r_thresh,
        shape=shape,
        settings=dict(
            qdamp=qdamp,
            polydispersity=polydispersity,
            fit_min=fit_min,
            noise_range=noise_range,
            window=window,
            threshold_factor=threshold_factor,
            persistence=persistence,
        ),
    )
    if np.isnan(r_thresh):
        result.status = (
            "The RMS never stays below the threshold before the noise "
            "range; the coherence length may exceed the data range."
        )
    if shape == "none":
        return result
    envelope = get_envelope(shape, polydispersity)
    if fit_max is None:
        fit_max = noise_range[0]
        if not np.isnan(r_thresh):
            fit_max = min(fit_max, max(1.4 * r_thresh, fit_min + 6 * window))
    result.settings["fit_max"] = fit_max
    fit_mask = (r >= fit_min) & (r <= fit_max)
    p0 = SHAPES[shape].p0_factor * (
        fit_max if np.isnan(r_thresh) else r_thresh
    )
    try:
        fit = fit_envelope(
            r[fit_mask],
            rms[fit_mask],
            envelope,
            qdamp,
            noise_rms,
            p0=p0,
            bounds=(2 * dr, 10 * r[-1]),
            step=window_points(r, window),
        )
    except RuntimeError as err:
        result.status = f"Envelope fit failed: {err}"
        return result
    result.fit = fit
    result.model_curve = model_rms(
        r, fit.amplitude, fit.param, envelope, qdamp, noise_rms
    )
    result.envelope_curve = (
        fit.amplitude * envelope(r, fit.param) * qdamp_envelope(r, qdamp)
    )
    below = np.flatnonzero((r >= fit_min) & (result.model_curve <= threshold))
    if len(below):
        result.r_model = float(r[below[0]])
    if fit.unconstrained:
        result.status = (
            "The data do not constrain the envelope parameter (upper "
            "bound reached or relative error > 50 %): the domains exceed "
            "the Qdamp/noise limit, or Qdamp is wrong."
        )
    return result
