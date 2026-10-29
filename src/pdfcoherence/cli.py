"""Command-line interface: ``pdfcoherence <file-or-folder> <qdamp>``."""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

from pdfcoherence.analysis import analyze
from pdfcoherence.envelopes import SHAPES
from pdfcoherence.io import (
    find_pdf_files,
    load_pdf,
    summary_row,
    write_csv,
    write_latex,
)
from pdfcoherence.plotting import plot_result
from pdfcoherence.version import __version__

EPILOG = """examples:
  pdfcoherence sample.gr 0.03
      sphere envelope, results in ./processed/
  pdfcoherence data/ 0.01 --shape lognormal --polydispersity 0.3 -o out/
      batch-process every .gr/.cgr/.dat in data/
  pdfcoherence data/ 0.01 --shape exponential --fit-min 20
      exponential decay, e.g. for disordered or porous frameworks
  pdfcoherence sim.cgr 0.01 --noise-level 0.05
      simulated, noise-free data need an imposed noise floor

The reported r_model is the largest distance at which structural
signal is detectable above the noise. It is a lower bound of the
structural coherence and depends on counting statistics, Qmax and Qdamp.
"""


def build_parser():
    """Build the argument parser.

    Returns
    -------
    argparse.ArgumentParser
        The parser of the ``pdfcoherence`` command.
    """
    parser = argparse.ArgumentParser(
        prog="pdfcoherence",
        description=(
            "Estimate the real-space coherence length of a material from "
            "its pair distribution function G(r) without a structural "
            "model. Writes one figure per PDF plus a summary CSV and "
            "LaTeX table."
        ),
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "target",
        type=Path,
        help="PDF file (.gr, .cgr, .dat) or folder of PDF files.",
    )
    parser.add_argument(
        "qdamp",
        type=float,
        help=(
            "Instrumental Qdamp in 1/A, refined on a calibrant (e.g. Ni "
            "or CeO2) measured with the same setup, e.g. 0.03. Use 0 for "
            "simulated data without instrumental damping."
        ),
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=Path("processed"),
        help="Output folder for figures and tables (default: processed).",
    )
    parser.add_argument(
        "-s",
        "--shape",
        choices=["none", *SHAPES],
        default="sphere",
        help=(
            "Envelope model. 'sphere': single diameter D; 'lognormal': "
            "spheres with a lognormal size distribution (mean D, width "
            "from --polydispersity); 'exponential': exp(-r/xi) decay; "
            "'none': only the model-free r_thresh (default: sphere)."
        ),
    )
    parser.add_argument(
        "--polydispersity",
        type=float,
        default=0.0,
        help=(
            "Relative standard deviation of the number-weighted diameter "
            "distribution for --shape lognormal, e.g. 0.3 for 30%% "
            "(default: 0)."
        ),
    )
    parser.add_argument(
        "--fit-min",
        type=float,
        default=15.0,
        help=(
            "Lower r-limit (A) of the fit and threshold scan. Place it "
            "beyond intra-cluster/unit-cell distances but well below the "
            "particle size (default: 15)."
        ),
    )
    parser.add_argument(
        "--fit-max",
        type=float,
        default=None,
        help=(
            "Upper r-limit (A) of the fit (default: 1.4 x r_thresh, "
            "capped at the start of the noise range)."
        ),
    )
    parser.add_argument(
        "--noise-min",
        type=float,
        default=None,
        help=(
            "Lower r-limit (A) of the noise-floor range. It must contain "
            "no structural signal (default: start of the last 20%% of "
            "the data)."
        ),
    )
    parser.add_argument(
        "--noise-max",
        type=float,
        default=None,
        help="Upper r-limit (A) of the noise-floor range (default: r_max).",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=None,
        help=(
            "RMS of white noise (units of G) added in quadrature to the "
            "rolling RMS and used as noise floor instead of the estimate. "
            "Needed for noise-free simulated data, e.g. 0.05."
        ),
    )
    parser.add_argument(
        "-w",
        "--window",
        type=float,
        default=10.0,
        help="Width (A) of the rolling RMS window (default: 10).",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=1.15,
        help=(
            "Detection threshold as a multiple of the noise floor. "
            "Narrow windows need larger values (default: 1.15)."
        ),
    )
    parser.add_argument(
        "-p",
        "--persistence",
        type=float,
        default=15.0,
        help=(
            "Distance (A) the measured RMS must stay below the threshold "
            "to define r_thresh (default: 15)."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return parser


def _noise_range(args, r):
    if args.noise_min is None and args.noise_max is None:
        return None
    lower = args.noise_min if args.noise_min is not None else r[0]
    upper = args.noise_max if args.noise_max is not None else r[-1]
    return lower, upper


def run(argv=None):
    """Run the analysis on a file or folder.

    Parameters
    ----------
    argv : list of str, optional
        The command-line arguments; ``sys.argv[1:]`` by default.

    Returns
    -------
    list of dict
        The summary row of each processed file.
    """
    args = build_parser().parse_args(argv)
    try:
        files = find_pdf_files(args.target)
    except (FileNotFoundError, NotADirectoryError) as err:
        sys.exit(f"Error: {err}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in files:
        print(f"Processing {path.name}")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                r, gr = load_pdf(path)
                result = analyze(
                    r,
                    gr,
                    args.qdamp,
                    shape=args.shape,
                    polydispersity=args.polydispersity,
                    fit_min=args.fit_min,
                    fit_max=args.fit_max,
                    noise_range=_noise_range(args, r),
                    noise_level=args.noise_level,
                    window=args.window,
                    threshold_factor=args.threshold,
                    persistence=args.persistence,
                )
        except ValueError as err:
            print(f"  Skipped: {err}")
            rows.append(summary_row(path.name, error=err))
            continue
        plot_result(
            result, args.out_dir / f"{path.stem}_coherence_analysis.png"
        )
        if result.status != "Success":
            print(f"  Note: {result.status}")
        rows.append(summary_row(path.name, result))
    write_csv(rows, args.out_dir / "summary_coherence_report.csv")
    write_latex(rows, args.shape, args.out_dir / "summary_coherence_table.tex")
    _print_summary(rows)
    print(f"\nFigures and tables written to {args.out_dir}")
    return rows


def main():
    """Entry point of the ``pdfcoherence`` command."""
    run()


def _print_summary(rows):
    header = (
        f"{'File':<40} {'r_thresh/A':>11} {'r_model/A':>10} {'Param/A':>16}"
    )
    print("\n" + header + "\n" + "-" * len(header))
    for row in rows:
        param = (
            "N/A"
            if np.isnan(row["param"])
            else f"{row['param']:.1f} +- {row['param_err']:.1f}"
        )
        print(
            f"{row['file']:<40} {row['r_thresh']:>11.1f} "
            f"{row['r_model']:>10.1f} {param:>16}"
        )


if __name__ == "__main__":
    main()
