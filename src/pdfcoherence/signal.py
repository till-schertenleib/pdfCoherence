"""Rolling RMS, noise floor and threshold crossing of a PDF."""

import warnings

import numpy as np


def grid_step(r):
    """Return the step of a uniform r-grid.

    Parameters
    ----------
    r : numpy.ndarray
        The real-space distances in Angstrom.

    Returns
    -------
    float
        The grid spacing in Angstrom.
    """
    steps = np.diff(r)
    if len(steps) == 0 or not np.allclose(steps, steps[0], rtol=1e-3):
        raise ValueError(
            "The r-grid is not uniformly spaced (or has fewer than two "
            "points). Re-grid the PDF onto a uniform r-grid, e.g. with "
            "PDFgetX3 or numpy.interp, before the analysis."
        )
    return float(steps[0])


def window_points(r, width):
    """Return the odd number of grid points spanning a window.

    Parameters
    ----------
    r : numpy.ndarray
        The real-space distances in Angstrom.
    width : float
        The window width in Angstrom.

    Returns
    -------
    int
        The number of points, at least 3 and always odd so the window
        is centred.
    """
    k = max(3, int(round(width / grid_step(r))))
    return k + 1 - k % 2


def rolling_rms(r, gr, width):
    """Root-mean-square of G(r) in a centred sliding window.

    Near the ends of the data the window is truncated and the mean is
    taken over the points actually inside it.

    Parameters
    ----------
    r : numpy.ndarray
        The real-space distances in Angstrom.
    gr : numpy.ndarray
        The PDF values G(r).
    width : float
        The window width in Angstrom.

    Returns
    -------
    numpy.ndarray
        The rolling RMS on the same grid as ``r``.
    """
    kernel = np.ones(window_points(r, width))
    sum_sq = np.convolve(gr**2, kernel, mode="same")
    count = np.convolve(np.ones_like(gr), kernel, mode="same")
    return np.sqrt(sum_sq / count)


def default_noise_range(r, fraction=0.2):
    """Return the last ``fraction`` of the r-range.

    Parameters
    ----------
    r : numpy.ndarray
        The real-space distances in Angstrom.
    fraction : float, optional
        The fraction of the r-range used for the noise estimate.

    Returns
    -------
    tuple of float
        The lower and upper limit of the noise range in Angstrom.
    """
    return float(r[-1] - fraction * (r[-1] - r[0])), float(r[-1])


def estimate_noise_rms(r, gr, r_min, r_max):
    """RMS of G(r) in a high-r range assumed to contain only noise.

    Parameters
    ----------
    r : numpy.ndarray
        The real-space distances in Angstrom.
    gr : numpy.ndarray
        The PDF values G(r).
    r_min : float
        The lower limit of the noise range in Angstrom.
    r_max : float
        The upper limit of the noise range in Angstrom.

    Returns
    -------
    float
        The noise RMS in the units of G(r).
    """
    mask = (r >= r_min) & (r <= r_max)
    if np.count_nonzero(mask) < 10:
        raise ValueError(
            f"The noise range {r_min:.1f}-{r_max:.1f} A contains fewer "
            f"than 10 points (data cover {r[0]:.1f}-{r[-1]:.1f} A). "
            "Choose a noise range inside the data."
        )
    noise_rms = float(np.sqrt(np.mean(gr[mask] ** 2)))
    half = np.count_nonzero(mask) // 2
    lower, upper = gr[mask][:half], gr[mask][half:]
    if np.sqrt(np.mean(lower**2)) > 1.25 * np.sqrt(np.mean(upper**2)):
        warnings.warn(
            f"The RMS of G(r) still decreases inside the noise range "
            f"{r_min:.1f}-{r_max:.1f} A, so it probably contains "
            "structural signal and the noise floor is overestimated. "
            "Move the noise range to higher r if the data allow it.",
            stacklevel=2,
        )
    return noise_rms


def find_threshold_crossing(r, rms, threshold, r_min, r_max, persistence):
    """First r where the rolling RMS falls below a threshold and stays
    there.

    Parameters
    ----------
    r : numpy.ndarray
        The real-space distances in Angstrom.
    rms : numpy.ndarray
        The rolling RMS of G(r).
    threshold : float
        The threshold in the units of G(r).
    r_min : float
        The distance in Angstrom where the scan starts.
    r_max : float
        The distance in Angstrom where the scan stops.
    persistence : float
        The distance in Angstrom over which the RMS must stay below the
        threshold.

    Returns
    -------
    float
        The crossing distance in Angstrom, or NaN if the RMS never stays
        below the threshold inside the scan range.
    """
    n_consecutive = max(3, int(round(persistence / grid_step(r))))
    mask = (r >= r_min) & (r <= r_max)
    below = (rms[mask] <= threshold).astype(int)
    if len(below) < n_consecutive:
        return np.nan
    runs = np.convolve(below, np.ones(n_consecutive, dtype=int), "valid")
    hits = np.flatnonzero(runs == n_consecutive)
    return float(r[mask][hits[0]]) if len(hits) else np.nan
