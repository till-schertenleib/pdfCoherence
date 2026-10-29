"""Log-space fit of the rolling RMS with an envelope model."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

from pdfcoherence.envelopes import qdamp_envelope


def model_rms(r, amplitude, param, envelope, qdamp, noise_rms):
    """Expected rolling RMS for an envelope on top of a noise floor.

    Parameters
    ----------
    r : numpy.ndarray
        The real-space distances in Angstrom.
    amplitude : float
        The scale factor A of the structural signal.
    param : float
        The envelope parameter in Angstrom.
    envelope : Callable
        The envelope function ``envelope(r, param)``.
    qdamp : float
        The instrumental damping parameter in inverse Angstrom.
    noise_rms : float
        The noise floor in the units of G(r).

    Returns
    -------
    numpy.ndarray
        ``sqrt((A * f(r) * B(r))**2 + noise_rms**2)``.
    """
    signal = amplitude * envelope(r, param) * qdamp_envelope(r, qdamp)
    return np.sqrt(signal**2 + noise_rms**2)


@dataclass
class EnvelopeFit:
    """Result of an envelope fit.

    Attributes
    ----------
    amplitude : float
        The refined scale factor A.
    param : float
        The refined envelope parameter in Angstrom.
    param_err : float
        The one-sigma statistical uncertainty of ``param`` in Angstrom.
    unconstrained : bool
        True if ``param`` ended on its upper bound or its relative
        uncertainty exceeds 50 %, i.e. the data do not constrain it.
    """

    amplitude: float
    param: float
    param_err: float
    unconstrained: bool


def fit_envelope(r, rms, envelope, qdamp, noise_rms, p0, bounds, step):
    """Fit ``log(rms)`` with ``log(model_rms)``.

    Only every ``step``-th point is used, because neighbouring points of
    a rolling RMS are strongly correlated; fitting all of them would
    underestimate the parameter uncertainty by roughly ``sqrt(step)``.

    Parameters
    ----------
    r : numpy.ndarray
        The real-space distances of the fit range in Angstrom.
    rms : numpy.ndarray
        The rolling RMS on ``r``.
    envelope : Callable
        The envelope function ``envelope(r, param)``.
    qdamp : float
        The instrumental damping parameter in inverse Angstrom.
    noise_rms : float
        The noise floor in the units of G(r).
    p0 : float
        The starting value of the envelope parameter in Angstrom.
    bounds : tuple of float
        The lower and upper bound of the envelope parameter in Angstrom.
    step : int
        The number of grid points between fitted points, typically the
        number of points in one RMS window.

    Returns
    -------
    EnvelopeFit
        The refined amplitude and envelope parameter.
    """
    r_fit, log_rms_fit = r[::step], np.log(rms[::step])
    if len(r_fit) < 4:
        raise ValueError(
            f"Only {len(r_fit)} independent points are in the fit range. "
            "Lower fit_min, raise fit_max or reduce the window width."
        )

    def log_model(r_eval, ln_amplitude, param):
        rms_eval = model_rms(
            r_eval, np.exp(ln_amplitude), param, envelope, qdamp, noise_rms
        )
        return np.log(np.maximum(rms_eval, 1e-300))

    ln_a0 = np.log(max(rms[0], noise_rms))
    popt, pcov = curve_fit(
        log_model,
        r_fit,
        log_rms_fit,
        p0=[ln_a0, np.clip(p0, *bounds)],
        bounds=([-50.0, bounds[0]], [50.0, bounds[1]]),
    )
    param_err = np.sqrt(pcov[1, 1]) if np.isfinite(pcov[1, 1]) else np.nan
    at_bound = np.isclose(popt[1], bounds[1], rtol=1e-3)
    return EnvelopeFit(
        amplitude=float(np.exp(popt[0])),
        param=float(popt[1]),
        param_err=float(param_err),
        unconstrained=bool(at_bound or not param_err < 0.5 * popt[1]),
    )
