import csv
from enum import Enum
from pathlib import Path
import typer
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from diffpy.utils.parsers.loaddata import loadData

app = typer.Typer(help="Batch process pair distribution functions to extract coherence lengths and export LaTeX/CSV tables.")


class ShapeModel(str, Enum):
    none = "none"
    spherical = "spherical"
    exponential = "exponential"


def f_spherical(r: np.ndarray, D: float) -> np.ndarray:
    ratio = np.clip(r / D, 0.0, 1.0)
    val = 1.0 - 1.5 * ratio + 0.5 * (ratio ** 3)
    return np.where(r <= D, val, 0.0)


def f_exponential(r: np.ndarray, xi: float) -> np.ndarray:
    return np.exp(-r / xi)


def compute_rolling_rms(r: np.ndarray, gr: np.ndarray, window_size: float) -> np.ndarray:
    dr = r[1] - r[0]
    k = max(3, int(round(window_size / dr)))
    rolling_var = np.convolve(gr**2, np.ones(k) / k, mode='same')
    return np.sqrt(rolling_var)


def estimate_noise_rms(r: np.ndarray, gr: np.ndarray, noise_min: float, noise_max: float) -> tuple[float, float, float]:
    r_max = r[-1]
    if r_max < noise_min:
        eff_min = r_max - 0.15 * (r_max - r[0])
        eff_max = r_max
    else:
        eff_min = noise_min
        eff_max = min(noise_max, r_max)

    noise_mask = (r >= eff_min) & (r <= eff_max)
    if not np.any(noise_mask):
        raise ValueError("Selected noise range contains no data points.")

    noise_rms = float(np.sqrt(np.mean(gr[noise_mask]**2)))
    return noise_rms, eff_min, eff_max


def find_forward_threshold_cutoff(
    r: np.ndarray, 
    rolling_rms: np.ndarray, 
    threshold_rms: float, 
    fit_min: float, 
    eff_n_min: float, 
    persistence_length: float = 15.0
) -> float:
    dr = r[1] - r[0]
    n_consecutive = max(3, int(round(persistence_length / dr)))
    
    scan_mask = (r >= fit_min) & (r <= eff_n_min)
    r_scan = r[scan_mask]
    rms_scan = rolling_rms[scan_mask]

    is_below = rms_scan <= threshold_rms

    for i in range(len(is_below) - n_consecutive):
        if np.all(is_below[i : i + n_consecutive]):
            return float(r_scan[i])
            
    return float(r_scan[-1])


def process_single_pdf(
    pdf_file: Path,
    qdamp: float,
    out_dir: Path,
    shape: ShapeModel,
    fit_min: float,
    noise_min: float,
    noise_max: float,
    window_size: float,
    threshold_factor: float,
    persistence: float
) -> dict:
    data = loadData(str(pdf_file))
    r = data[:, 0]
    gr_meas = data[:, 1]

    noise_rms, eff_n_min, eff_n_max = estimate_noise_rms(r, gr_meas, noise_min, noise_max)
    rolling_rms = compute_rolling_rms(r, gr_meas, window_size)
    threshold_rms = threshold_factor * noise_rms

    r_threshold = find_forward_threshold_cutoff(
        r=r,
        rolling_rms=rolling_rms,
        threshold_rms=threshold_rms,
        fit_min=fit_min,
        eff_n_min=eff_n_min,
        persistence_length=persistence
    )

    fit_max = min(eff_n_min, max(100.0, r_threshold * 1.4))
    r_fit_mask = (r >= fit_min) & (r <= fit_max)
    r_fit = r[r_fit_mask]
    log_rms_fit = np.log(rolling_rms[r_fit_mask])
    b_inst = np.exp(-0.5 * (r * qdamp)**2)

    envelope_curve = None
    fitted_rms_curve = None
    r_model_cutoff = np.nan
    param_val = np.nan
    param_err = np.nan

    if shape == ShapeModel.spherical:
        def log_rms_sphere(r_eval, ln_A, D):
            A = np.exp(ln_A)
            f = f_spherical(r_eval, D)
            b = np.exp(-0.5 * (r_eval * qdamp)**2)
            model_val = np.sqrt((A * f * b)**2 + noise_rms**2)
            return np.log(np.maximum(model_val, 1e-12))

        initial_ln_A = np.log(max(rolling_rms[r >= fit_min][0], 1e-3))
        p0 = [initial_ln_A, r_threshold]
        bounds = ([-20.0, 5.0], [10.0, 400.0])

        try:
            popt, pcov = curve_fit(log_rms_sphere, r_fit, log_rms_fit, p0=p0, bounds=bounds)
            ln_A_fit, D_fit = popt
            A_fit = np.exp(ln_A_fit)
            param_val = float(D_fit)
            param_err = float(np.sqrt(pcov[1, 1])) if pcov is not None else 0.0

            fitted_rms_curve = np.exp(log_rms_sphere(r, ln_A_fit, D_fit))
            envelope_curve = A_fit * f_spherical(r, D_fit) * b_inst

            model_below = np.where((r >= fit_min) & (fitted_rms_curve <= threshold_rms))[0]
            if len(model_below) > 0:
                r_model_cutoff = float(r[model_below[0]])
        except Exception as e:
            typer.secho(f"Model fit failed for {pdf_file.name}: {e}", fg=typer.colors.YELLOW)

    elif shape == ShapeModel.exponential:
        def log_rms_exp(r_eval, ln_A, xi):
            A = np.exp(ln_A)
            f = f_exponential(r_eval, xi)
            b = np.exp(-0.5 * (r_eval * qdamp)**2)
            model_val = np.sqrt((A * f * b)**2 + noise_rms**2)
            return np.log(np.maximum(model_val, 1e-12))

        initial_ln_A = np.log(max(rolling_rms[r >= fit_min][0], 1e-3))
        p0 = [initial_ln_A, 25.0]
        bounds = ([-20.0, 2.0], [10.0, 200.0])

        try:
            popt, pcov = curve_fit(log_rms_exp, r_fit, log_rms_fit, p0=p0, bounds=bounds)
            ln_A_fit, xi_fit = popt
            A_fit = np.exp(ln_A_fit)
            param_val = float(xi_fit)
            param_err = float(np.sqrt(pcov[1, 1])) if pcov is not None else 0.0

            fitted_rms_curve = np.exp(log_rms_exp(r, ln_A_fit, xi_fit))
            envelope_curve = A_fit * f_exponential(r, xi_fit) * b_inst

            model_below = np.where((r >= fit_min) & (fitted_rms_curve <= threshold_rms))[0]
            if len(model_below) > 0:
                r_model_cutoff = float(r[model_below[0]])
        except Exception as e:
            typer.secho(f"Model fit failed for {pdf_file.name}: {e}", fg=typer.colors.YELLOW)

    # Plot generation
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    display_limit = min(r[-1], max(120.0, r_threshold * 1.6))
    vis_mask = r <= display_limit

    # Top Panel
    ax1.plot(r, gr_meas, color='black', alpha=0.7, linewidth=1.0, label=r'$G_{meas}(r)$')
    ax1.axhline(0, color='gray', linestyle=':', linewidth=0.8)

    if envelope_curve is not None:
        ax1.plot(r, envelope_curve, color='crimson', linestyle='--', linewidth=1.5, label=r'Envelope $\pm A \cdot f(r) \cdot B(r)$')
        ax1.plot(r, -envelope_curve, color='crimson', linestyle='--', linewidth=1.5)
        ax1.fill_between(r, -envelope_curve, envelope_curve, color='crimson', alpha=0.07)

    ax1.axvline(r_threshold, color='darkorange', linestyle='--', linewidth=1.5, label=rf'Empirical Cutoff: {r_threshold:.1f} $\mathrm{{\AA}}$')
    if not np.isnan(r_model_cutoff):
        ax1.axvline(r_model_cutoff, color='royalblue', linestyle=':', linewidth=1.5, label=rf'Model Crossing: {r_model_cutoff:.1f} $\mathrm{{\AA}}$')

    y_min, y_max = np.min(gr_meas[vis_mask]), np.max(gr_meas[vis_mask])
    pad = (y_max - y_min) * 0.1
    ax1.set_ylim(y_min - pad, y_max + pad)
    ax1.set_ylabel(r'$G(r)$ ($\mathrm{\AA}^{-2}$)', fontsize=12)
    ax1.set_title(f"Coherence Length Analysis: {pdf_file.name}", fontsize=13)
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)

    # Bottom Panel
    ax2.plot(r, rolling_rms, color='teal', linewidth=1.4, label=rf'Rolling RMS ({window_size:.1f} $\mathrm{{\AA}}$ window)')
    ax2.axhline(noise_rms, color='gray', linestyle='--', linewidth=1.2, label=rf'Noise Floor $\sigma_{{noise}}$ ({noise_rms:.2e} $\mathrm{{\AA}}^{{-2}}$)')
    ax2.axhline(threshold_rms, color='darkorange', linestyle='--', linewidth=1.2, label=rf'Threshold ({threshold_factor:.2f}x $\sigma_{{noise}}$)')

    if fitted_rms_curve is not None:
        ax2.plot(r, fitted_rms_curve, color='crimson', linewidth=2.0, label='Log-Space Analytical Fit')

    ax2.axvline(r_threshold, color='darkorange', linestyle='--', linewidth=1.5)
    if not np.isnan(r_model_cutoff):
        ax2.axvline(r_model_cutoff, color='royalblue', linestyle=':', linewidth=1.5)

    ax2.set_xlabel(r'$r$ ($\mathrm{\AA}$)', fontsize=12)
    ax2.set_ylabel(r'RMS ($\mathrm{\AA}^{-2}$)', fontsize=12)
    ax2.set_yscale('log')
    ax2.set_xlim(r[0], display_limit)
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    out_plot_path = out_dir / f"{pdf_file.stem}_coherence_analysis.png"
    plt.savefig(out_plot_path, dpi=300)
    plt.close()

    return {
        "file": pdf_file.name,
        "empirical_threshold": r_threshold,
        "model_crossing": r_model_cutoff,
        "param_val": param_val,
        "param_err": param_err,
        "noise_rms": noise_rms,
        "status": "Success"
    }


def write_summary_reports(results: list[dict], out_dir: Path, shape: ShapeModel):
    # 1. Export CSV
    csv_path = out_dir / "summary_coherence_report.csv"
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "File",
            "Empirical_Threshold_A",
            "Model_Crossing_A",
            "Fitted_Parameter_A",
            "Parameter_Error_A",
            "Noise_RMS_A-2"
        ])
        for r in results:
            writer.writerow([
                r["file"],
                f"{r['empirical_threshold']:.2f}" if not np.isnan(r["empirical_threshold"]) else "N/A",
                f"{r['model_crossing']:.2f}" if not np.isnan(r["model_crossing"]) else "N/A",
                f"{r['param_val']:.2f}" if not np.isnan(r["param_val"]) else "N/A",
                f"{r['param_err']:.2f}" if not np.isnan(r["param_err"]) else "N/A",
                f"{r['noise_rms']:.4e}" if not np.isnan(r["noise_rms"]) else "N/A"
            ])

    # 2. Export LaTeX Table
    tex_path = out_dir / "summary_coherence_table.tex"
    param_header = "$D$ (\\AA)" if shape == ShapeModel.spherical else ("$\\xi$ (\\AA)" if shape == ShapeModel.exponential else "Parameter (\\AA)")

    tex_lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Summary of pair distribution function coherence lengths and instrumental noise analysis.}",
        r"  \label{tab:pdf_coherence_summary}",
        r"  \begin{tabular}{lcccc}",
        r"    \toprule",
        rf"    File & $r_{{\mathrm{{threshold}}}}$ (\AA) & $r_{{\mathrm{{model}}}}$ (\AA) & {param_header} & $\sigma_{{\mathrm{{noise}}}}$ (\AA$^{{-2}}$) \\",
        r"    \midrule"
    ]

    for r in results:
        # Escape LaTeX special characters in file names
        sanitized_filename = r["file"].replace("_", r"\_").replace("%", r"\%")
        emp_str = f"{r['empirical_threshold']:.1f}" if not np.isnan(r["empirical_threshold"]) else "--"
        mod_str = f"{r['model_crossing']:.1f}" if not np.isnan(r["model_crossing"]) else "--"
        
        if not np.isnan(r["param_val"]):
            param_str = f"{r['param_val']:.1f} \\pm {r['param_err']:.1f}"
        else:
            param_str = "--"

        noise_str = f"{r['noise_rms']:.2e}" if not np.isnan(r["noise_rms"]) else "--"

        tex_lines.append(rf"    {sanitized_filename} & {emp_str} & {mod_str} & ${param_str}$ & {noise_str} \\")

    tex_lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}"
    ])

    with open(tex_path, mode="w") as f:
        f.write("\n".join(tex_lines))

    typer.secho(f"\nSaved CSV Report:   {csv_path}", fg=typer.colors.GREEN, bold=True)
    typer.secho(f"Saved LaTeX Table:  {tex_path}", fg=typer.colors.GREEN, bold=True)


@app.command()
def analyze(
    target_path: Path = typer.Argument(..., help="Path to input PDF file (.gr/.cgr) or folder containing PDFs"),
    qdamp: float = typer.Argument(..., help="Instrumental Qdamp parameter (in Å^-1)"),
    out_dir: Path = typer.Option(Path("."), "--out-dir", "-o", help="Output directory folder"),
    shape: ShapeModel = typer.Option(ShapeModel.spherical, "--shape", "-s", help="Structural envelope model"),
    fit_min: float = typer.Option(15.0, "--fit-min", help="Minimum r (Å) for fitting"),
    noise_min: float = typer.Option(300.0, "--noise-min", help="Minimum r for noise estimation (Å)"),
    noise_max: float = typer.Option(400.0, "--noise-max", help="Maximum r for noise estimation (Å)"),
    window_size: float = typer.Option(10.0, "--window-size", "-w", help="Rolling RMS window width (Å)"),
    threshold_factor: float = typer.Option(1.15, "--threshold", "-t", help="Threshold factor over noise RMS"),
    persistence: float = typer.Option(15.0, "--persistence", "-p", help="Distance in Å the signal must remain below threshold")
):
    """
    Evaluates pair distribution functions either for a single file or in batch mode 
    across all files in a folder, exporting figures, a summary CSV, and a LaTeX table.
    """
    if not target_path.exists():
        typer.secho(f"Error: Path '{target_path}' does not exist.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Detect batch mode vs single file
    if target_path.is_dir():
        pdf_extensions = {".gr", ".cgr", ".dat"}
        file_list = sorted([p for p in target_path.iterdir() if p.suffix.lower() in pdf_extensions and p.is_file()])
        if not file_list:
            typer.secho(f"No PDF files (.gr, .cgr, .dat) found in '{target_path}'.", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        typer.secho(f"Found {len(file_list)} PDF files in directory. Processing batch...", fg=typer.colors.CYAN, bold=True)
    else:
        file_list = [target_path]

    results = []

    for pdf_file in file_list:
        typer.echo(f"Processing: {pdf_file.name}")
        try:
            res = process_single_pdf(
                pdf_file=pdf_file,
                qdamp=qdamp,
                out_dir=out_dir,
                shape=shape,
                fit_min=fit_min,
                noise_min=noise_min,
                noise_max=noise_max,
                window_size=window_size,
                threshold_factor=threshold_factor,
                persistence=persistence
            )
            results.append(res)
        except Exception as e:
            typer.secho(f"Error processing {pdf_file.name}: {e}", fg=typer.colors.RED)
            results.append({
                "file": pdf_file.name,
                "empirical_threshold": np.nan,
                "model_crossing": np.nan,
                "param_val": np.nan,
                "param_err": np.nan,
                "noise_rms": np.nan,
                "status": f"Failed: {e}"
            })

    # Export summary tables when batch processing
    if len(file_list) > 1 or target_path.is_dir():
        write_summary_reports(results, out_dir, shape)

    # Print summary to terminal
    typer.secho("\n--- Summary Table ---", fg=typer.colors.CYAN, bold=True)
    header = f"{'File':<35} {'r_thresh (Å)':<15} {'r_model (Å)':<15} {'Param (Å)':<20}"
    typer.echo(header)
    typer.echo("-" * len(header))
    for r in results:
        param_str = f"{r['param_val']:.1f} ± {r['param_err']:.1f}" if not np.isnan(r['param_val']) else "N/A"
        emp_str = f"{r['empirical_threshold']:.1f}" if not np.isnan(r['empirical_threshold']) else "N/A"
        mod_str = f"{r['model_crossing']:.1f}" if not np.isnan(r['model_crossing']) else "N/A"
        typer.echo(f"{r['file']:<35} {emp_str:<15} {mod_str:<15} {param_str:<20}")


if __name__ == "__main__":
    app()