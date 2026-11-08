from pathlib import Path
p = Path.home() / "sphere_noise_paper/paper.tex"
s = p.read_text()

edits = []

# --- Introduction ---
edits.append((
"""Prior work on $SO(3)$ in generative models treats rotation as a
symmetry to be respected; we invert the question: \\emph{what if rotation
is the corruption itself?}""",
"""Prior work on $SO(3)$ in generative
models~\\citep{thomas2018tensor,cohen2018spherical} treats rotation as a
symmetry to be respected. Diffusion models on
$SO(3)$~\\citep{leach2022diffusion,yim2023unified} study the reverse
problem: denoising an $SO(3)$-valued signal. We invert the question:
\\emph{what if rotation is the corruption itself?}"""
))

# --- Setup: spherical harmonics ---
edits.append((
"""with $\\dim\\mathcal{V}_L = d_L = (L+1)^2$.""",
"""with $\\dim\\mathcal{V}_L = d_L = (L+1)^2$~\\citep{driscoll1994computing}."""
))

# --- Prop 1 proof: Euler ---
edits.append((
"""The ZYZ parametrization is a local
diffeomorphism onto $SO(3)$ away from the polar singularities.""",
"""The ZYZ parametrization~\\citep{kanatani1990group,ivanic1996rotation} is a
local diffeomorphism onto $SO(3)$ away from the polar singularities."""
))

# --- Prop 2: CRB (claim) ---
edits.append((
"""\\end{equation}
\\end{proposition}
\\begin{proof}
Since $\\mathcal{V}_L$ is $SO(3)$-invariant,""",
"""\\end{equation}~\\citep{marzetta1995,crb_sh_doa}
\\end{proposition}
\\begin{proof}
Since $\\mathcal{V}_L$ is $SO(3)$-invariant~\\citep{sloan1994funk,schuh1996least},"""
))

# --- Prop 3 proof: torsor ---
edits.append((
"""$SO(3)$ acts on $\\mathcal{V}_L$ by pullback $(D\\cdot f)(x)=f(D^\\top x)$.""",
"""$SO(3)$ acts on $\\mathcal{V}_L$ by pullback $(D\\cdot f)(x)=f(D^\\top x)$~\\citep{torsorcnn2024}."""
))

# --- Section Contrast with Diffusion: new intro paragraph ---
edits.append((
"""\\section{Contrast with Diffusion}
\\begin{table}[ht]""",
"""\\section{Contrast with Diffusion}
The practice of applying diffusion to rotations has recently gained
traction on Lie groups and manifolds~\\citep{leach2022diffusion,yim2023unified,radm2025,equivdiffa5_2025},
where the noise is itself drawn from the group. We stress that the
rotation we consider is not a distributional sample: it is a *single
deterministic* element $R_\\Sigma\\in SO(3)$ applied to $f$. The ambiguity is
purely group-theoretic, not probabilistic. Table~\\ref{tab:contrast}
summarizes the difference.
\\begin{table}[ht]"""
))

# --- Future work: equivariant + Sobolev ---
edits.append((
"""\\paragraph{Equivariant networks.} Tensor-field networks should encode the
$SO(3)$-torsor structure by construction; this is the natural next step.""",
"""\\paragraph{Equivariant networks.} Tensor-field
networks~\\citep{thomas2018tensor,cohen2018spherical} should encode the
$SO(3)$-torsor structure by construction; this is the natural next step."""
))
edits.append((
"""For Sobolev-regular $f$ the truncation error is $O(L^{-s})$; numerical
sweep is planned.""",
"""For Sobolev-regular $f$ the truncation error is $O(L^{-s})$~\\citep{lychagin2021metric};
a numerical sweep is planned."""
))

# --- Append label to contrast table ---
edits.append((
"""\\caption{Two noise models compared.}
\\end{table}""",
"""\\caption{Two noise models compared.}
\\label{tab:contrast}
\\end{table}"""
))

applied, skipped = 0, 0
for old, new in edits:
    if new.split("~\\cite")[0][:30] in s and old not in s:
        print(f"SKIP (already patched): {old[:50]!r}")
        skipped += 1
    elif old in s:
        s = s.replace(old, new, 1)
        applied += 1
    else:
        print(f"WARN not found: {old[:70]!r}")
        skipped += 1

p.write_text(s)
print(f"\nApplied {applied} edits, skipped {skipped}.")
print("Citations now used:", s.count("\\cite{"))
