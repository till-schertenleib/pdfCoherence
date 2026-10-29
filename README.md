# pdfcoherence

`pdfcoherence` estimates how far in real space structural correlations can
be seen in an experimental pair distribution function (PDF, $G(r)$). It does
this **without a structural model** (no `.cif` file is needed).

It works on PDFs of nanoparticles, porous frameworks (e.g. MOFs) and
disordered materials. You can process one file or a whole folder, from the
command line or from Python.

> **What the number means.** The main output, $r_{\mathrm{model}}$, is the
> largest distance at which structural signal can still be told apart from
> noise. It is a _detection limit_, the "maximum observable $r$" of the PDF.
> It is **not** the particle size. Read [Limitations](#5-limitations) before
> you interpret it.

---

## Installation

```bash
conda create -n pdfcoherence_env -c conda-forge python=3.12 \
    --file requirements/conda.txt --file requirements/tests.txt
conda activate pdfcoherence_env
pip install -e . --no-deps
```

## Quick start

### Command line

```bash
# One file, sphere envelope, Qdamp = 0.03 1/A; results go to ./processed/
pdfcoherence sample.gr 0.03

# Every .gr/.cgr/.dat file in a folder, polydisperse spheres (30 %)
pdfcoherence data/ 0.01 --shape lognormal --polydispersity 0.3 -o processed/

# Disordered or porous material: exponential decay, fit from 20 A
pdfcoherence data/ 0.01 --shape exponential --fit-min 20

# Noise-free simulated PDF: impose a noise floor
pdfcoherence simulated.cgr 0.01 --noise-level 0.05
```

Run `pdfcoherence --help` to see all options.

For each PDF the command writes:

- a figure, `<name>_coherence_analysis.png`;
- one row in `summary_coherence_report.csv`;
- one row in `summary_coherence_table.tex`.

### Python

```python
from pdfcoherence.analysis import analyze
from pdfcoherence.io import load_pdf
from pdfcoherence.plotting import plot_result

r, gr = load_pdf("data/uio66_10nm_30percent_polydisp_noisy.cgr")
result = analyze(r, gr, qdamp=0.01, shape="lognormal", polydispersity=0.3)
print(result.r_thresh, result.r_model, result.fit.param, result.status)
plot_result(result, "uio66_10nm.png")
```

---

## 1. Signal model

### Measured PDF

The measured PDF is modelled as the bulk-like structural signal multiplied
by two envelopes, plus additive noise:

$$G_{\mathrm{meas}}(r) = A\, f(r)\, B(r)\, G_{\mathrm{bulk}}(r) + \epsilon(r)$$

The terms are:

- $A$ is a scale factor.
- $G_{\mathrm{bulk}}(r)$ is the PDF of the ideal, infinite structure. For a
  crystal its oscillation amplitude stays roughly constant with $r$.
- $f(r)$ describes how the structural coherence decays. Its options are
  listed in [Envelope models](#envelope-models) below.
- $B(r)$ is the damping caused by the finite $Q$-resolution of the
  instrument.
- $\epsilon(r)$ is noise, mainly from counting statistics.

### Instrumental damping $B(r)$

$Q_{\mathrm{damp}}$ (in $\mathrm{\AA}^{-1}$) has the same definition as in
PDFgui (Farrow et al., 2007):

$$B(r) = \exp\left[-\tfrac{1}{2}\left(r\, Q_{\mathrm{damp}}\right)^2\right]$$

$Q_{\mathrm{damp}}$ is **fixed**. Refine it beforehand on a calibrant (e.g.
Ni, CeO₂ or Si) measured with the same setup.

### Envelope models

| `--shape`     | $f(r)$                                                          | Refined parameter                |
| ------------- | --------------------------------------------------------------- | -------------------------------- |
| `sphere`      | $1 - \frac{3r}{2D} + \frac{r^3}{2D^3}$ for $r \le D$, else $0$  | Diameter $D$                     |
| `lognormal`   | $\langle D^3 f_{\mathrm{sph}}(r;D)\rangle / \langle D^3\rangle$ | Mean diameter $\langle D\rangle$ |
| `exponential` | $\exp(-r/\xi)$                                                  | Correlation length $\xi$         |
| `none`        | no fit                                                          | (only $r_{\mathrm{thresh}}$)     |

**`sphere`.** This is the characteristic function of a homogeneous sphere
(Guinier & Fournet, 1955; Kodama et al., 2006). It is the same function as
the spherical-particle envelope in PDFgui (Masadeh et al., 2007).

**`lognormal`.** This envelope describes an ensemble of spheres whose
diameters follow a lognormal _number_ distribution.

- The polydispersity is $p = \sigma_D / \langle D \rangle$. It is set with
  `--polydispersity`; it is not refined.
- The PDF is an average per atom. Each particle therefore contributes in
  proportion to its number of atoms, $\propto D^3$, so large particles
  dominate the tail.
- Weighting a lognormal by $D^3$ gives another lognormal, with
  $s^2 = \ln(1+p^2)$ and $\mu_w = \ln\langle D\rangle - s^2/2 + 3s^2$.
- Because $f_{\mathrm{sph}}$ is a polynomial in $1/D$, the ensemble average
  has an exact closed form:

$$
f(r) = M_0 - \tfrac{3r}{2} M_{-1} + \tfrac{r^3}{2} M_{-3},
\qquad M_k(r) = e^{k\mu_w + k^2 s^2/2}\,
\Phi\!\left(\frac{\mu_w + k s^2 - \ln r}{s}\right)
$$

Here $\Phi$ is the standard normal cumulative distribution function. For
$p = 0$ the envelope reduces to `sphere`. The test suite checks the closed
form against direct numerical integration.

**`exponential`.** This is the Debye–Anderson–Brumberger correlation
function of a random two-phase medium (Debye et al., 1957). Here it is used
as an _ad hoc_ envelope for materials whose order decays gradually rather
than ending at a sharp boundary.

The refined parameters of different shapes are **not** interchangeable. For
example, near $r = 0$ the sphere envelope falls with slope $-3/(2D)$ and the
exponential with slope $-1/\xi$, so $\xi \approx 2D/3$ there and not $D$.

---

## 2. Noise floor and rolling RMS

### Noise floor $\sigma_{\mathrm{noise}}$

$\sigma_{\mathrm{noise}}$ is the RMS of $G(r)$ in a high-$r$ range that
contains no structural signal. By default this is the last 20 % of the data;
change it with `--noise-min` and `--noise-max`.

$$\sigma_{\mathrm{noise}} = \sqrt{\frac{1}{N}\sum_{r_i \in \text{noise range}} G_{\mathrm{meas}}(r_i)^2}$$

Why the noise floor is the same at all $r$:

- $G(r)$ is a sine Fourier transform of $F(Q) = Q[S(Q)-1]$.
- If the errors in $F(Q)$ are uncorrelated, the variance of $G(r)$ is
  $\propto \sum_i \sigma_F^2(Q_i)\sin^2(Q_i r)$.
- For $r \gg \pi/Q_{\max}$ this sum is essentially independent of $r$, so
  one constant noise level applies everywhere.
- The noise in $G(r)$ is correlated over $\Delta r \approx \pi/Q_{\max}$.

### Rolling RMS

The structural signal is summarised by the RMS of $G(r)$ in a centred
sliding window of width $W$ (`--window`, default 10 Å):

$$\mathrm{RMS}_{\mathrm{meas}}(r) = \sqrt{\frac{1}{K}\sum_{|r'-r|\le W/2} G_{\mathrm{meas}}(r')^2}$$

$K$ is the number of grid points inside the window. At the ends of the
data, the window is truncated and $K$ is reduced accordingly.

Signal and uncorrelated noise add in quadrature, so the model for the
rolling RMS is:

$$\mathrm{RMS}_{\mathrm{model}}(r) = \sqrt{\left[A\, f(r)\, B(r)\right]^2 + \sigma_{\mathrm{noise}}^2}$$

### Threshold $k$ and window $W$

A window contains about $N_{\mathrm{ind}} \approx W Q_{\max}/\pi$
independent noise values. The RMS of pure noise therefore scatters with a
relative standard deviation of about $1/\sqrt{2N_{\mathrm{ind}}}$.

Example: with $W = 10$ Å and $Q_{\max} = 22\ \mathrm{\AA}^{-1}$,
$N_{\mathrm{ind}} \approx 70$ and the scatter is about 8.5 %. The default
threshold $k = 1.15$ then sits about $1.8\sigma$ above the noise floor.

- Narrower windows or lower $Q_{\max}$ need a larger $k$.
- The persistence condition (below) further suppresses false crossings.

---

## 3. Fit

The fit minimises residuals in logarithmic space. This gives every decade of
signal decay the same weight, instead of letting the strong low-$r$ peaks
dominate:

$$\chi^2(A, p) = \sum_{r_j} \left[\ln \mathrm{RMS}_{\mathrm{meas}}(r_j) - \ln \mathrm{RMS}_{\mathrm{model}}(r_j; A, p)\right]^2$$

- **Free and fixed parameters.** $A$ and the shape parameter $p$ ($D$,
  $\langle D\rangle$ or $\xi$) are refined. $Q_{\mathrm{damp}}$ and
  $\sigma_{\mathrm{noise}}$ stay fixed.
- **Independent points only.** Only points one window apart ($r_j$ spaced
  by $W$) are fitted. Neighbouring rolling-RMS values are strongly
  correlated, and fitting all of them would shrink the reported
  uncertainty by roughly $\sqrt{K}$.
- **Lower limit (`--fit-min`, default 15 Å).** Set it beyond
  intra-cluster or unit-cell distances, so the fit describes the decay of
  coherence rather than local structure. For small particles it must stay
  well below the particle size.
- **Upper limit (`--fit-max`).** The default is $1.4\, r_{\mathrm{thresh}}$,
  capped at the start of the noise range.

---

## 4. Outputs and how to read them

| Output                    | Definition                                                                                                                                                                                     |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| $r_{\mathrm{thresh}}$     | The first $r \ge$ `fit-min` where the _measured_ rolling RMS drops below $k\,\sigma_{\mathrm{noise}}$ and stays below for `--persistence` (default 15 Å). Model-free.                          |
| $r_{\mathrm{model}}$      | Where the _fitted_ $\mathrm{RMS}_{\mathrm{model}}$ drops below $k\,\sigma_{\mathrm{noise}}$. It uses the whole decay curve, so it is less sensitive to local noise than $r_{\mathrm{thresh}}$. |
| Param ± err               | The refined $D$, $\langle D\rangle$ or $\xi$, with its _statistical_ one-sigma error from the fit.                                                                                             |
| $\sigma_{\mathrm{noise}}$ | The noise floor, in the units of $G(r)$.                                                                                                                                                       |
| Status                    | "Success" or a warning: no threshold crossing, fit failed, or a parameter the data do not constrain.                                                                                           |

### Why Param is usually larger than $r_{\mathrm{model}}$

1. **Signal reaches the noise before $D$.** $f(r)$ falls to zero only at
   $r = D$, and $B(r)$ suppresses the signal further. The product
   therefore drops into the noise well before $D$. Param extrapolates to
   where the signal _would_ vanish with perfect resolution and no noise.
2. **Polydispersity.** Each particle is weighted by its atom count
   ($\propto D^3$), so the largest particles of a size distribution set
   the tail of the PDF. A single-diameter `sphere` fitted to a
   polydisperse sample returns a $D$ above the number-weighted mean. The
   `lognormal` shape accounts for this, but only if the polydispersity
   you give it is correct.

### Worked example

These are the simulated UiO-66 PDFs in `data/`, nominally 10 nm and
100 nm with 30 % polydispersity. They were analysed with
$Q_{\mathrm{damp}} = 0.01\ \mathrm{\AA}^{-1}$ and the default settings.
All lengths are in Å.

| File          | Shape                  | $r_{\mathrm{thresh}}$ | $r_{\mathrm{model}}$ | Param                            |
| ------------- | ---------------------- | --------------------- | -------------------- | -------------------------------- |
| 10 nm, noisy  | `sphere`               | 102.1                 | 103.1                | $D$ = 136.8 ± 3.5                |
| 10 nm, noisy  | `lognormal`, $p$ = 0.3 | 102.1                 | 107.8                | $\langle D\rangle$ = 104.6 ± 3.7 |
| 10 nm, noisy  | `exponential`          | 102.1                 | 115.2                | $\xi$ = 43.4 ± 3.4               |
| 100 nm, noisy | any                    | 250.7                 | 247 ± 1              | not constrained                  |

What the example shows:

- **$r_{\mathrm{model}}$ depends only weakly on the shape.** It changes by up
  to 12 % between shapes (103–115 Å). Param varies by a factor of three.
- **The 100 nm sample is limited by $Q_{\mathrm{damp}}$ and noise, not by
  size.** The signal disappears near 250 Å, far below $D$ = 1000 Å, and
  the size is flagged as unconstrained.
- **Noise-free data are refused** unless `--noise-level` is given.

---

## 5. Limitations

This is a deliberately simple, model-free estimate. Keep the following in
mind.

- **$r_{\mathrm{model}}$ is a detection limit and a lower bound.** Atom pairs
  exist beyond it, but their signal is below the noise. The value depends on

  - counting statistics (longer counting gives a larger $r_{\mathrm{model}}$),
  - $Q_{\max}$, which sets the noise level and correlation,
  - $Q_{\mathrm{damp}}$,
  - the threshold $k$, the window $W$ and the persistence length.

  Compare samples only when they were measured and processed the same way.

- **$Q_{\mathrm{damp}}$ must be right.** With a wrong value the fit is wrong
  too. For domains larger than a few $1/Q_{\mathrm{damp}}$, the size and
  $Q_{\mathrm{damp}}$ cannot be separated, and Param is then reported as
  unconstrained.
- **Param depends on the model.** $D$, $\langle D\rangle$ and $\xi$ answer
  different questions. Quote the shape model, and for `lognormal` the
  assumed polydispersity, with every value.
- **The quoted uncertainties are statistical only.** They do not include
  the choice of shape, $Q_{\mathrm{damp}}$, the fit range or the noise
  range, and these usually dominate.
- **The noise range must be free of structural signal.** A warning is
  printed when the RMS still decreases inside it. If the coherence extends
  to the end of the data, $r_{\mathrm{thresh}}$ cannot be determined.
- **Noise-free (simulated) PDFs** have no meaningful noise floor. Use
  `--noise-level` to add one in quadrature.
- **Effects that are not modelled:**
  - $r$-dependent peak broadening (Qbroad, correlated motion), which lowers
    peak heights and so the RMS;
  - strain, stacking faults and other extended defects;
  - termination ripples and background-subtraction errors;
  - non-spherical or anisotropic domains;
  - multiple phases;
  - surface relaxation and ligand or solvent contributions.
- **A uniform $r$-grid is required.**

---

## References

- P. Debye, H. R. Anderson, H. Brumberger, "Scattering by an Inhomogeneous
  Solid. II. The Correlation Function and Its Application", _J. Appl.
  Phys._ **28**, 679–683 (1957).
  [doi:10.1063/1.1722830](https://doi.org/10.1063/1.1722830)
- A. Guinier, G. Fournet, _Small-Angle Scattering of X-Rays_, Wiley,
  New York (1955).
- K. Kodama, S. Iikubo, T. Taguchi, S. Shamoto, "Finite size effects of
  nanoparticles on the atomic pair distribution functions", _Acta Cryst._
  A **62**, 444–453 (2006).
  [doi:10.1107/S0108767306034635](https://doi.org/10.1107/S0108767306034635)
- A. S. Masadeh, E. S. Božin, C. L. Farrow, G. Paglia, P. Juhás,
  S. J. L. Billinge, A. Karkamkar, M. G. Kanatzidis, "Quantitative
  size-dependent structure and strain determination of CdSe nanoparticles
  using atomic pair distribution function analysis", _Phys. Rev. B_ **76**,
  115413 (2007).
  [doi:10.1103/PhysRevB.76.115413](https://doi.org/10.1103/PhysRevB.76.115413)
- C. L. Farrow, P. Juhás, J. W. Liu, D. Bryndin, E. S. Božin, J. Bloch,
  Th. Proffen, S. J. L. Billinge, "PDFfit2 and PDFgui: computer programs
  for studying nanostructure in crystals", _J. Phys.: Condens. Matter_
  **19**, 335219 (2007).
  [doi:10.1088/0953-8984/19/33/335219](https://doi.org/10.1088/0953-8984/19/33/335219)

## License

MIT. See [LICENSE](LICENSE).
