# Noor on the Sphere — Rotation as Invertible Noise

> **Kung-fu opening.** استاد می‌گوید: روی تپه برو، پاها را خم کن، تنه‌ی درخت را تصور کن و بغلش کن.
> همه‌چیز روی یک کره اتفاق می‌افتد — و همه‌ی حرکات یک چرخش‌اند.

## 1. Geometry

$$S^{2}=\{\,p\in\mathbb{R}^{3}:\ \lVert p\rVert=1\,\}$$

## 2. Meteor strikes (data events)

$$m_{i}=(\mathbf{p}_{i},\boldsymbol{\omega}_{i},\Delta t_{i})\in S^{2}\times\mathbb{R}^{3}\times\mathbb{R}_{+}$$

Angular momentum, axis and angle:

$$\mathbf{L}_{i}=\mathbf{p}_{i}\times\boldsymbol{\omega}_{i},\qquad
\hat{\mathbf{n}}_{i}=\frac{\mathbf{L}_{i}}{\lVert\mathbf{L}_{i}\rVert},\qquad
\theta_{i}=\lVert\mathbf{L}_{i}\rVert\,\Delta t_{i}$$

Rodrigues rotation:

$$R_{i}=\exp\!\bigl(\theta_{i}[\hat{\mathbf{n}}_{i}]_{\times}\bigr)
= I + \sin\theta_{i}\,[\hat{\mathbf{n}}_{i}]_{\times}
   + (1-\cos\theta_{i})\,[\hat{\mathbf{n}}_{i}]_{\times}^{2}$$

Total composed rotation (order matters):

$$R_{\Sigma}=R_{N}\cdots R_{2}R_{1}\in SO(3)$$

**Key geometric fact** (this is the "kung-fu" point):
$\dim SO(3)=3$, so no matter how many strikes, the whole effect is **3 numbers**:

$$R_{\Sigma}=R_{z}(\alpha)R_{y}(\beta)R_{z}(\gamma)$$

## 3. Data field on the sphere

$$f(\mathbf{x})=\sum_{l=0}^{L}\sum_{m=-l}^{l}a_{lm}\,Y_{lm}(\mathbf{x})$$

## 4. Observation = field after cumulative rotation (the "noise")

$$g(\mathbf{x})=f\bigl(R_{\Sigma}^{\top}\mathbf{x}\bigr)+\varepsilon(\mathbf{x})$$

## 5. Extraction (the diffusion-inverse)

**Known \(R_{\Sigma}\):**

$$\hat{f}(\mathbf{x})=g\!\left(R_{\Sigma}\,\mathbf{x}\right)$$

**Unknown \(R_{\Sigma}\) (diffusion-style):**

$$(\hat{f},\hat{R}_{\Sigma})=\arg\min_{f,\,R}\ \mathbb{E}_{\mathbf{x}}\!\left[\bigl(f(R^{\top}\mathbf{x})-g(\mathbf{x})\bigr)^{2}\right]
\ \ \text{s.t.}\ f=\sum a_{lm}Y_{lm}$$

## 6. The gauge theorem (core of the paper)

Rotation "noise" is **invertible** but has a **stabilizer subgroup**:

$$\mathrm{Stab}(f)=\{R\in SO(3):\ R\!\cdot\! f=f\}$$

- \(L=0\)  ⇒ \(\mathrm{Stab}=SO(3)\) (unidentifiable: isotropic field looks the same in every direction)
- \(L=1\)  ⇒ \(\mathrm{Stab}=SO(2)\) (continuous gauge — dipole can be rotated away)
- \(L\geq 2\) ⇒ \(\mathrm{Stab}=\text{finite subgroup of }SO(3)\) (discrete gauge)

So unlike Gaussian diffusion, the forward "noise" here **does not destroy information**;
the real hard part is **breaking the gauge**.

## 7. Experiments (run.sh)

| # | What | What it shows |
|---|------|---------------|
| A | 500 random strikes on \(S^2\) | closure in \(SO(3)\) + 3-number Euler compression |
| B | recover \(f\) from \(g\) with known \(R_\Sigma\) | error tracks noise (invertibility) |
| C | recover \(f\) and \(R_\Sigma\) jointly (\(L=2\)) | diffusion-style optimization, up to gauge |
| D | noise sweep | \(\text{err}\sim \sigma\) |
| E | Euler error vs \(N\) | 3 numbers regardless of \(N\) |

Plots in `plots/`, raw numbers in `results/results.json`.


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24062931.svg)](https://doi.org/10.5281/zenodo.24062931)
