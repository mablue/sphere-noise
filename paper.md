# Rotation as Invertible Noise:
# Geometry, Stabilizers, and Multi-view Identification on the Sphere

## Abstract
We study a generative model in which "noise" is a random rotation
$R_\Sigma \in SO(3)$ applied to a band-limited field $f$ on $S^2$. We show:

1. The cumulative effect of any number of "strikes" on $S^2$ collapses to a
   single rotation with **3 degrees of freedom** (500x compression,
   $10^{-16}$ error).
2. With $R_\Sigma$ known, recovery is information-theoretically optimal
   (slope = 1.04, matching Cramér-Rao).
3. With $R_\Sigma$ **unknown**, a single view leaves an **$SO(3)$-torsor**
   gauge — three continuous dimensions of ambiguity that no amount of
   data can resolve (Proposition 3).
4. With **two views** of the same $f$, identification reduces to the
   **stabilizer** $\mathrm{Stab}(f) \subseteq SO(3)$ — finite or continuous,
   depending on the symmetry class of $f$ (Propositions 4, 5).

This contrasts sharply with Gaussian diffusion: rotations preserve
information but induce a group-theoretic gauge, not information destruction.

---

## 1. Introduction

The kung-fu image: **everything happens on a sphere, every movement is
one rotation of that sphere**. In generative modeling terms: replace additive
Gaussian noise with **group action noise**.

Prior work on $SO(3)$ in generative models (e.g., tensor-field networks,
$SE(3)$-diffusion) treats rotation as a *symmetry* to be respected.
We invert the question: **what if rotation is the corruption itself?**

---

## 2. Setup and notation

$$S^2 = \{p\in\mathbb{R}^3 : \|p\|=1\}$$

Band-limited field on the sphere with $L$-truncation:

$$f(x) = \sum_{l=0}^{L}\sum_{m=-l}^{l} a_{lm}\, Y_{lm}(x),\qquad f\in\mathcal{V}_L$$

where $\mathcal{V}_L = \mathrm{span}\{Y_l^m\}_{l\le L}$ has dimension
$d_L = (L+1)^2$.

**Strike model.** A strike is a triple $m_i = (p_i,\omega_i,\Delta t_i)$,
producing rotation

$$R_i = \exp(\theta_i[\hat n_i]_\times),\quad \theta_i = \|p_i\times\omega_i\|\Delta t_i$$

**Composed rotation:** $R_\Sigma = R_N\cdots R_1 \in SO(3)$.

**Observation** (invertible "noise"):

$$g(x) = f(R_\Sigma^\top x) + \varepsilon(x)$$

---

## 3. Proposition 1 (Closure and 3-DOF compression)

> **Claim.** For any $N$ and any sequence of strikes, $R_\Sigma\in SO(3)$
> and can be represented by 3 Euler parameters.
>
> **Proof.** $SO(3)$ is a closed subgroup of $GL_3(\mathbb{R})$ of dimension 3,
> hence every product of rotations is itself a rotation. ZYZ Euler
> parametrization $R = R_z(\alpha)R_y(\beta)R_z(\gamma)$ is a local
> diffeomorphism onto $SO(3)\setminus\{\text{poles}\}$.
>
> **Empirical.** $\|R_\Sigma R_\Sigma^\top-I\|=3.3\times10^{-16}$ over 500 strikes;
> Euler reconstruction error $4.3\times10^{-16}$.

**Figure 1**: Fibonacci-distributed strike points on $S^2$.

---

## 4. Proposition 2 (Optimal recovery with known $R$)

> **Claim.** If $R_\Sigma$ is known and observations are $N$ Fibonacci
> samples $x_j$, the least-squares estimator
> $\hat f(R_\Sigma^\top x) = g(x)$ satisfies
> $$\mathbb{E}\|\hat f - f\| / \|f\| \approx \sigma\|f\|^{-1}\sqrt{d_L/N}.$$
>
> **Proof.** $g(R_\Sigma^{-1}\cdot)$ belongs to $\mathcal{V}_L$ by
> $SO(3)$-invariance of $\mathcal{V}_L$. Least-squares estimation of $d_L$
> coefficients from $N$ i.i.d. Gaussian observations has variance
> $\sigma^2 d_L / N$; the relative error follows by Cauchy–Schwarz.

**Empirical (Fig 12)**: 7 noise levels, slopes $1.039$, matches CRB.

---

## 5. Proposition 3 (SO(3)-torsor on a single view)

> **Claim.** Given a single observation $g = f\circ R_\Sigma^{-1} + \varepsilon$,
> the pair $(f,R_\Sigma)$ is identifiable only up to
> $$(f, R_\Sigma) \sim (f\circ D^{-1},\ R_\Sigma\cdot D)\qquad \forall D\in SO(3).$$
>
> **Proof.** $SO(3)$ acts on $\mathcal{V}_L$ by pullback
> $(D\cdot f)(x) = f(D^\top x)$. This action is *transitive on the orbit*
> $\{f\circ D^{-1}:D\in SO(3)\}$, and the residual $\|f(R^\top\cdot) - g\|$
> is invariant under the joint transform $R\mapsto RD$, $f\mapsto f\circ D^{-1}$.
> Hence no functional of $g$ can distinguish members of the orbit.

**Empirical (V2, Table 3)**: least-squares fit residual is
$1.22\times10^{-3}$ for $R_\Sigma$ chosen true, identity, and two random
rotations; coefficient norms identical to $10^{-15}$.

---

## 6. Proposition 4 (Stabilizer for known field)

> **Claim.** With $f$ known, $R_\Sigma$ is identifiable up to the stabilizer
> $$\mathrm{Stab}(f) = \{D\in SO(3): f\circ D = f\}.$$
>
> **Structure.** For $f = Y_l^0$ (zonal):
> $$\mathrm{Stab}(Y_l^0) = SO(2)_z\ \cup\ \{R_\pi(\hat n):\hat n\perp\hat z\}.$$
>
> **Proof.** $Y_l^0(x) = K P_l(x_z)$, so $(D\cdot Y_l^0)(x) = K P_l((D^\top x)_z)$.
> Equality with $Y_l^0$ for all $x$ forces $(D^\top x)_z = \pm x_z$, i.e. $D^\top\hat z = \pm\hat z$. The $+$ branch is $SO(2)_z$; the $-$ branch is a
> $\pi$-rotation about an axis in the $xy$-plane.

**Empirical (V3, Fig 13)**: 18 recovered $\Delta$'s split exactly into
these two components — continuous angles on the $SO(2)_z$ branch,
discrete $|\Delta|=\pi$ on the $\pi_{xy}$ branch.

---

## 7. Proposition 5 (Multi-view identification)

> **Claim.** Given two views $g_1=f\circ R_1^{-1}$, $g_2=f\circ R_2^{-1}$,
> the *relative* rotation is identifiable up to $\mathrm{Stab}(f)$:
> $$R_1^{-1}R_2 \in SO(3)/\mathrm{Stab}(f).$$
>
> **Proof.** Let $D$ be any rotation satisfying $g_1(D^{-1}\cdot)=g_2$. Then
> $f(R_1^{-1}D^{-1}\cdot)=f(R_2^{-1}\cdot)$ as functions, i.e.
> $R_1^{-1}D^{-1}R_2 \in \mathrm{Stab}(f)$. Solving for $D$ gives
> $D = R_2\Delta R_1^{-1}$ with $\Delta\in\mathrm{Stab}(f)$.

**Empirical (Phase5)**:
- Generic $f$ ($L=3$): recovery error $2.4\times10^{-7}$ → **unique**.
- Zonal $f=Y_2^0$: all 40 recovered $D$'s differ by $\Delta\in\mathrm{Stab}(Y_2^0)$.

**Figure 14**: all $\Delta$ values cluster in $\mathrm{Stab}(f)$.

---

## 8. Hierarchy of ambiguities

| Regime | Ambiguity group | Dimension |
|---|---|---|
| 1 view, any $f$ | $SO(3)$ | 3 (continuous) |
| $\geq2$ views, generic $f$ | $\{I\}$ | 0 |
| $\geq2$ views, $f=Y_2^0$ | $SO(2)\cup\pi_{xy}$ | 1 + finite |
| $\geq2$ views, finite-sym $f$ | finite subgroup | 0 |

This table is our main take-away message.

---

## 9. Contrast with diffusion

| Aspect | Gaussian diffusion | Rotation noise (this paper) |
|---|---|---|
| Information destroyed | Yes (heat equation) | No (rotation invertible) |
| Ambiguity source | Noise realizations | Group gauge |
| Optimal recovery | Score matching | Linear algebra (Prop 2) |
| Identification | Approximate | **Exact up to Stab(f)** (Prop 4) |
| Symmetry handling | Irrelevant | Central (Prop 3, 4, 5) |

---

## 10. Limitations and future work

- The field $f$ is assumed band-limited ($\in\mathcal{V}_L$). For
  non-band-limited $f$, the truncation error is $O(L^{-s})$ for Sobolev
  regularity $s$; numerical experiments planned.
- End-to-end learning of $R_\Sigma$ from data (rather than optimization)
  is future work; our Phase 2 MLP (Exp F) confirms the gauge gap: it learns
  the field but not the rotation, exactly as Proposition 3 predicts.
- Equivariant networks (Tensor Field Networks) should encode the
  $SO(3)$-torsor structure by construction.

---

## References

(to be filled)
