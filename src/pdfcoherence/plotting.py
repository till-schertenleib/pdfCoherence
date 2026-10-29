"""Two-panel figure of a coherence-length analysis."""

import matplotlib as mpl
import numpy as np
from matplotlib.figure import Figure

CM = 1 / 2.54
BLUE = (0 / 255, 114 / 255, 178 / 255)
VERM = (213 / 255, 94 / 255, 0 / 255)
GRAY = "0.5"
STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.6,
    "lines.linewidth": 0.8,
}


def plot_result(result, path, display_max=None):
    """Plot G(r) with the fitted envelope and the rolling RMS.

    Panel a shows G(r) with the envelope ``+-A f(r) B(r)``; panel b
    shows the rolling RMS, the fitted model, the noise floor and the
    threshold on a logarithmic axis. No data are shifted or offset.

    Parameters
    ----------
    result : CoherenceResult
        The analysis result.
    path : pathlib.Path
        The output image file.
    display_max : float, optional
        The upper r-limit of the plot in Angstrom. By default
        ``1.6 * r_thresh`` (at least 120 A), capped at the data range.
    """
    r, s = result.r, result.settings
    if display_max is None:
        reference = result.r_thresh if np.isfinite(result.r_thresh) else r[-1]
        display_max = min(r[-1], max(120.0, 1.6 * reference))
    visible = r <= display_max
    width = 9 * CM
    with mpl.rc_context(STYLE):
        fig = Figure(figsize=(width, width), layout="constrained")
        ax_g, ax_rms = fig.subplots(
            2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 2]}
        )
        ax_g.plot(r, result.gr, color="black", lw=0.5, label="Data")
        if result.envelope_curve is not None:
            for sign in (1, -1):
                ax_g.plot(
                    r,
                    sign * result.envelope_curve,
                    color=BLUE,
                    ls="--",
                    label="Envelope" if sign == 1 else None,
                )
        ax_rms.plot(r, result.rms, color="black", label="Rolling RMS")
        if result.model_curve is not None:
            ax_rms.plot(r, result.model_curve, color=BLUE, label="Model")
        ax_rms.axhline(
            result.noise_rms, color=GRAY, ls=":", label="Noise floor"
        )
        ax_rms.axhline(
            result.threshold,
            color=GRAY,
            ls="--",
            label=f"Threshold ({s['threshold_factor']:.2f}"
            r"$\,\sigma_{\mathrm{noise}}$)",
        )
        for ax in (ax_g, ax_rms):
            if np.isfinite(result.r_thresh):
                ax.axvline(
                    result.r_thresh,
                    color=GRAY,
                    ls="-.",
                    label=(
                        rf"$r_{{\mathrm{{thresh}}}}$ = "
                        rf"{result.r_thresh:.1f} $\mathrm{{\AA}}$"
                    ),
                )
            if np.isfinite(result.r_model):
                ax.axvline(
                    result.r_model,
                    color=VERM,
                    ls="-",
                    label=(
                        rf"$r_{{\mathrm{{model}}}}$ = "
                        rf"{result.r_model:.1f} $\mathrm{{\AA}}$"
                    ),
                )
        g_vis = result.gr[visible]
        pad = 0.1 * (g_vis.max() - g_vis.min())
        ax_g.set_ylim(g_vis.min() - pad, g_vis.max() + pad)
        ax_g.set_ylabel(r"$G(r) \, / \, \mathrm{\AA}^{-2}$")
        ax_rms.set_yscale("log")
        ax_rms.set_xlim(r[0], display_max)
        ax_rms.set_xlabel(r"$r \, / \, \mathrm{\AA}$")
        ax_rms.set_ylabel(r"$\mathrm{RMS} \, / \, \mathrm{\AA}^{-2}$")
        for ax, letter in ((ax_g, "a"), (ax_rms, "b")):
            ax.legend(loc="upper right", frameon=False)
            ax.text(
                -0.16,
                1.0,
                letter,
                transform=ax.transAxes,
                fontweight="bold",
                va="top",
            )
        fig.savefig(path, dpi=300)
