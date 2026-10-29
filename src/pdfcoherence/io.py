"""Reading PDFs and writing summary reports."""

import csv
from pathlib import Path

import numpy as np
from diffpy.utils.parsers import load_data

from pdfcoherence.envelopes import SHAPES

PDF_SUFFIXES = {".gr", ".cgr", ".dat"}


def load_pdf(path):
    """Read r and G(r) from a PDF file (.gr, .cgr or two-column text).

    Parameters
    ----------
    path : pathlib.Path
        The path to the PDF file.

    Returns
    -------
    tuple of numpy.ndarray
        The r-grid in Angstrom and G(r).
    """
    data = load_data(Path(path))
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(
            f"Could not find two data columns (r, G) in '{path}'. Check "
            "that the file is a PDF with r in the first column and G(r) "
            "in the second."
        )
    return data[:, 0], data[:, 1]


def find_pdf_files(target):
    """List PDF files in a folder, or return a single file.

    Parameters
    ----------
    target : pathlib.Path
        The PDF file or folder of PDF files.

    Returns
    -------
    list of pathlib.Path
        The sorted PDF files.
    """
    target = Path(target)
    if target.is_file():
        return [target]
    files = sorted(
        p
        for p in target.iterdir()
        if p.is_file() and p.suffix.lower() in PDF_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(
            f"No PDF files ({', '.join(sorted(PDF_SUFFIXES))}) found in "
            f"'{target}'. Point to a PDF file or a folder containing them."
        )
    return files


def summary_row(name, result=None, error=None):
    """Collect the reported numbers of one analysis.

    Parameters
    ----------
    name : str
        The name of the analysed file.
    result : CoherenceResult, optional
        The analysis result; omit it for a failed analysis.
    error : str, optional
        The error message of a failed analysis.

    Returns
    -------
    dict
        The file name, r_thresh, r_model, param, param_err, noise_rms
        and status.
    """
    fit = result.fit if result is not None else None
    return {
        "file": name,
        "r_thresh": result.r_thresh if result else np.nan,
        "r_model": result.r_model if result else np.nan,
        "param": fit.param if fit else np.nan,
        "param_err": fit.param_err if fit else np.nan,
        "noise_rms": result.noise_rms if result else np.nan,
        "status": result.status if result else f"Failed: {error}",
    }


def _fmt(value, spec, missing):
    return missing if np.isnan(value) else format(value, spec)


def write_csv(rows, path):
    """Write the summary rows to a CSV file.

    Parameters
    ----------
    rows : list of dict
        The rows returned by ``summary_row``.
    path : pathlib.Path
        The output file.
    """
    with Path(path).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "File",
                "r_thresh_A",
                "r_model_A",
                "Param_A",
                "Param_err_A",
                "Noise_RMS_A-2",
                "Status",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["file"],
                    _fmt(row["r_thresh"], ".2f", "N/A"),
                    _fmt(row["r_model"], ".2f", "N/A"),
                    _fmt(row["param"], ".2f", "N/A"),
                    _fmt(row["param_err"], ".2f", "N/A"),
                    _fmt(row["noise_rms"], ".4e", "N/A"),
                    row["status"],
                ]
            )


def write_latex(rows, shape, path):
    """Write the summary rows as a LaTeX (booktabs) table.

    Parameters
    ----------
    rows : list of dict
        The rows returned by ``summary_row``.
    shape : str
        The envelope model, used for the parameter column header.
    path : pathlib.Path
        The output file.
    """
    symbol = SHAPES[shape].param_label if shape in SHAPES else "-"
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Real-space coherence lengths from the PDF.}",
        r"  \label{tab:pdf_coherence_summary}",
        r"  \begin{tabular}{lcccc}",
        r"    \toprule",
        r"    File & $r_{\mathrm{thresh}}$ / \AA"
        r" & $r_{\mathrm{model}}$ / \AA"
        rf" & ${symbol}$ / \AA"
        r" & $\sigma_{\mathrm{noise}}$ / \AA$^{-2}$ \\",
        r"    \midrule",
    ]
    for row in rows:
        name = row["file"].replace("_", r"\_").replace("%", r"\%")
        param = (
            "--"
            if np.isnan(row["param"])
            else f"${row['param']:.1f} \\pm {row['param_err']:.1f}$"
        )
        lines.append(
            f"    {name} & {_fmt(row['r_thresh'], '.1f', '--')} & "
            f"{_fmt(row['r_model'], '.1f', '--')} & {param} & "
            f"{_fmt(row['noise_rms'], '.2e', '--')} \\\\"
        )
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    Path(path).write_text("\n".join(lines))
