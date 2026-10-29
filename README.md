# PDF Coherence Length Analysis

This methodology extracts the structural coherence length $r_{coh}$ of a material from its experimental pair distribution function (PDF, $G(r)$) without requiring an atomistic structural model (`.cif` file).

---

## 1. Physical Definitions & Signal Formulation

### Structural Coherence Length ($r_{coh}$)

The coherence length $r_{coh}$ is the maximum distance in real space where correlated atom pairs exist within the sample domain. Beyond $r_{coh}$, structural peak oscillations vanish, leaving only statistical counting noise and Fourier artifacts.

### The Measured Signal Model

The measured pair distribution function $G_{meas}(r)$ is modeled as the product of the true structural correlation amplitude, a domain shape envelope, an instrumental resolution dampening envelope, and additive noise:

$$G_{meas}(r) = A \cdot f(r) \cdot B(r) + \epsilon(r)$$

* **Instrumental Dampening Envelope ($B(r)$):** Arises from finite instrument resolution in reciprocal space with parameter $Q_{damp}$ ($\mathrm{\AA}^{-1}$):

$$B(r) = \exp\left(-\frac{(r \cdot Q_{damp})^2}{2}\right)$$


* **Sample Shape Factor Envelope ($f(r)$):** Describes structural decay:
* **Spherical Domains:**

$$f(r; D) = 1 - \frac{3r}{2D} + \frac{r^3}{2D^3} \quad (r \le D; \ 0 \text{ for } r > D)$$


* **Disordered / Correlated Networks:**

$$f(r; \xi) = \exp\left(-\frac{r}{\xi}\right)$$




* **Baseline Counting Noise ($\sigma_{noise}$):** Determined directly from the flat baseline of uncorrected $G_{meas}(r)$ at high $r$ (e.g., 300–400 Å) where true structural signal is absent:

$$\sigma_{noise} = \sqrt{\frac{1}{N}\sum_{r_i \in \text{baseline}} G_{meas}(r_i)^2}$$



### The Rolling RMS Curve ($\text{RMS}_{meas}(r)$)

Instead of fitting individual oscillatory peaks, the algorithm computes the root-mean-square amplitude over a sliding boxcar window of width $W$ (typically 10 Å):

$$\text{RMS}_{meas}(r) = \sqrt{\frac{1}{K}\sum_{r' \in [r - W/2, \, r + W/2]} G_{meas}(r')^2}$$

Where $K$ is the number of grid pint within the boxcar window. Because physical signal and uncorrelated noise combine in quadrature, the theoretical RMS curve follows:

$$\text{RMS}_{model}(r; A, \text{Param}) = \sqrt{\left[ A \cdot f(r) \cdot B(r) \right]^2 + \sigma_{noise}^2}$$

Where $A$ is a scale factor that allows scaling the functions to $G_{meas}(r)$.


---

## 2. Optimization Function (Target Objective)

Linear least-squares fitting is heavily biased by high peak amplitudes at low $r$ (e.g., strong intra-cluster bonding in metal-organic frameworks at $r < 15\text{ \AA}$), which causes the optimizer to mischaracterize the subtle decay tail near the particle boundary.

To ensure equal relative weighting across all decades of signal decay, the algorithm minimizes residuals in **logarithmic space**:

$$\chi^2(A, \text{Param}) = \sum_{r_i \in [r_{min}, r_{max}]} \left[ \ln\left(\text{RMS}_{meas}(r_i)\right) - \ln\left(\text{RMS}_{model}(r_i; A, \text{Param})\right) \right]^2$$

* $Q_{damp}$ and $\sigma_{noise}$ remain fixed.
* Amplitude $A$ and structural parameter $\text{Param}$ ($D$ or $\xi$) are freely refined.
* $r_{min}$ is set beyond intra-unit-cell/cluster distances ($r \ge 15\text{--}20\text{ \AA}$) to ensure the fit models particle boundary decay rather than framework porosity.

---

## 3. Output Parameters

The report generates three distinct real-space distance parameters:

* `r_thresh` ($r_{\mathrm{threshold}}$): The empirical cutoff distance where the raw $\text{RMS}_{meas}(r)$ curve drops below the threshold level ($k \cdot \sigma_{noise}$, typically $1.15 \times \sigma_{noise}$) and **stays below it** continuously for a defined persistence length (e.g., 15 Å).
* `Param`: The refined mathematical parameter of the chosen envelope function:
* For `--shape spherical`: The fitted diameter $D$ (the point where the geometrical shape factor mathematically reaches zero, $f(D) = 0$).
* For `--shape exponential`: The correlation length $\xi$.


* `r_model`: The real-space distance where the **continuous analytical model curve** $\text{RMS}_{model}(r)$ drops below the detection threshold ($k \cdot \sigma_{noise}$).

---

## 4. Why `r_model` and `Param` Differ (and Why `r_model` is Superior)

In experimental data—especially with polydisperse nanoparticles—`Param` ($D$) is often substantially larger than `r_model`:

1. **Exponential Instrumental Suppression:**
Even if atoms exist out to distance $D$, the combined attenuation of geometric boundary decay $f(r)$ and instrumental decay $B(r)$ forces the physical signal into the noise floor $\sigma_{noise}$ well before reaching $D$. Beyond that crossing point, the theoretical curve is mathematically indistinguishable from a flat line at $\sigma_{noise}$.
2. **Volumetric Weighting ($D^3$ Effect):**
X-ray scattering intensity scales with volume fraction. In a polydisperse sample (e.g., 10 nm mean diameter with 30% polydispersity), the 15–18 nm particles produce four times more scattering signal than 8 nm particles. A monodisperse spherical shape function must expand $D$ outwards to accommodate this large-particle tail.

### Why `r_model` is the Most Robust Metric

* **`Param` ($D$) over-extrapolates:** It attempts to predict where the signal *would* reach absolute zero in an instrument with infinite resolution and zero noise.
* **`r_thresh` is vulnerable to noise spikes:** While forward scanning with persistence avoids single-point triggering, statistical counting noise can still create premature or delayed crossings.
* **`r_model` integrates the whole dataset:** By determining where the fitted regression curve drops below the experimental noise floor, `r_model` is immune to localized statistical noise fluctuations while correctly reporting the real, observable structural coherence limit of the sample.

---

## References

### Exponential Correlation Function 

P. Debye, H. R. Anderson, and H. Brumberger
"Scattering by an Inhomogeneous Solid. II. The Correlation Function and Its Application."
Journal of Applied Physics, 1957, 28(6), 679–683.
DOI: 10.1063/1.1722830

### Spherical Shape Function 

A. S. Masadeh, E. S. Božin, C. L. Farrow, G. Paglia, P. Juhás, S. J. L. Billinge, et al.
"Quantitative size-dependent structure and strain determination of CdSe nanoparticles using atomic pair distribution function analysis."
Physical Review B, 2007, 76(11), 115413.
DOI: 10.1103/PhysRevB.76.115413

A. Guinier and G. Fournet
Small-Angle Scattering of X-Rays.
John Wiley & Sons, New York, 1955.
