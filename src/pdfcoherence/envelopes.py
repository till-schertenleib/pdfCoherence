"""Envelope functions describing how the PDF signal decays with r."""

from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np
from scipy.special import ndtr


def qdamp_envelope(r, qdamp):
    """Gaussian damping caused by the finite Q-resolution of the
    instrument.

    Parameters
    ----------
    r : numpy.ndarray
        The real-space distances in Angstrom.
    qdamp : float
        The instrumental damping parameter in inverse Angstrom, as
        refined on a calibrant (e.g. in PDFgui).

    Returns
    -------
    numpy.ndarray
        The envelope ``exp(-(r * qdamp)**2 / 2)``.
    """
    return np.exp(-0.5 * (r * qdamp) ** 2)


def sphere(r, diameter):
    """Characteristic function of a homogeneous sphere.

    Parameters
    ----------
    r : numpy.ndarray
        The real-space distances in Angstrom.
    diameter : float
        The sphere diameter D in Angstrom.

    Returns
    -------
    numpy.ndarray
        The envelope ``1 - 3r/(2D) + r**3/(2D**3)`` for r <= D and zero
        beyond.
    """
    x = np.clip(r / diameter, 0.0, 1.0)
    return 1.0 - 1.5 * x + 0.5 * x**3


def lognormal_sphere(r, diameter, polydispersity):
    """Characteristic function of an ensemble of spheres with a
    lognormal number distribution of diameters.

    Each particle contributes in proportion to its number of atoms
    (proportional to D**3), so the envelope is
    ``<D**3 sphere(r, D)> / <D**3>``. The D**3-weighted lognormal is
    again lognormal, and the average is evaluated exactly through
    lognormal partial moments.

    Parameters
    ----------
    r : numpy.ndarray
        The real-space distances in Angstrom.
    diameter : float
        The number-weighted mean diameter in Angstrom.
    polydispersity : float
        The relative standard deviation of the number-weighted
        diameter distribution (e.g. 0.3 for 30 %). Zero reduces to
        ``sphere``.

    Returns
    -------
    numpy.ndarray
        The ensemble-averaged envelope.
    """
    if polydispersity <= 0:
        return sphere(r, diameter)
    s2 = np.log1p(polydispersity**2)
    s = np.sqrt(s2)
    mu_w = np.log(diameter) - 0.5 * s2 + 3.0 * s2
    with np.errstate(divide="ignore"):
        log_r = np.log(r)

    def partial_moment(k):
        return np.exp(k * mu_w + 0.5 * k**2 * s2) * ndtr(
            (mu_w + k * s2 - log_r) / s
        )

    return (
        partial_moment(0)
        - 1.5 * r * partial_moment(-1)
        + 0.5 * r**3 * partial_moment(-3)
    )


def exponential(r, xi):
    """Exponential decay of structural correlations.

    Parameters
    ----------
    r : numpy.ndarray
        The real-space distances in Angstrom.
    xi : float
        The correlation length in Angstrom.

    Returns
    -------
    numpy.ndarray
        The envelope ``exp(-r / xi)``.
    """
    return np.exp(-r / xi)


@dataclass(frozen=True)
class Shape:
    """A refinable envelope model.

    Attributes
    ----------
    func : Callable
        The envelope function ``func(r, param, **kwargs)``.
    param_label : str
        The LaTeX symbol of the refined parameter.
    p0_factor : float
        The starting value of the parameter relative to the empirical
        threshold distance.
    """

    func: Callable
    param_label: str
    p0_factor: float


SHAPES = {
    "sphere": Shape(sphere, "D", 1.0),
    "lognormal": Shape(lognormal_sphere, r"\langle D \rangle", 1.0),
    "exponential": Shape(exponential, r"\xi", 1.0 / 3.0),
}


def get_envelope(shape, polydispersity=0.0):
    """Return a one-parameter envelope ``f(r, param)`` for a shape.

    Parameters
    ----------
    shape : str
        The name of the shape, one of the keys of ``SHAPES``.
    polydispersity : float, optional
        The relative width of the diameter distribution, used only by
        the ``lognormal`` shape.

    Returns
    -------
    Callable
        The envelope function of ``r`` and the refined parameter.
    """
    if shape not in SHAPES:
        raise ValueError(
            f"Unknown shape '{shape}'. " f"Choose one of: {', '.join(SHAPES)}."
        )
    if shape == "lognormal":
        return partial(lognormal_sphere, polydispersity=polydispersity)
    return SHAPES[shape].func
