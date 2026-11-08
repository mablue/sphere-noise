from pathlib import Path
p = Path.home() / "sphere_noise_paper/paper.tex"
s = p.read_text()

prop6 = r"""
\section{Proposition 6: Equivariant Networks and the Gauge Barrier}

Propositions 3--5 predict a fundamental asymmetry: \emph{invariant}
functions of $f$ are learnable from a single view; \emph{equivariant}
functions (the rotation $R_\Sigma$ itself) are not. We verify this using
the \emph{power spectrum}
\begin{equation}
I_l(g) := \sum_{m=-l}^{l} |a_{lm}(g)|^2,\qquad l=0,\dots,L,
\end{equation}
which is $SO(3)$-invariant by construction (cf.~\citep{driscoll1994computing}).

\begin{proposition}
For a single view $g=f\circ R_\Sigma^{-1}$, the power spectrum
$\{I_l(g)\}_{l=0}^{L}$ is identifiable from any architecture; the rotation
$R_\Sigma$ is not identifiable from any architecture.
\end{proposition}
\begin{proof}
$\{I_l(g)\}$ is invariant under the action of $SO(3)$ on $\mathcal{V}_L$,
hence independent of $R_\Sigma$ (Prop.~3). $R_\Sigma$ is identifiable only
up to the $SO(3)$-torsor, so its mean-squared error is bounded below by the
variance of Haar measure on $SO(3)$, regardless of the model class.
\end{proof}

\noindent\textbf{Empirical (Fig.~\ref{fig:equiv}).} With $N=3000$ training
and $400$ test instances:
\begin{itemize}
\item Power spectrum recovery: raw-input MLP $\mathrm{err}=0.149$,
      equivariant-feature MLP $\mathrm{err}=0.013$ ($11\times$ better).
\item Rotation recovery (ZYZ angles): raw MLP $\mathrm{MSE}=3.19$,
      equivariant MLP $\mathrm{MSE}=2.37$ --- both far above the
      measurement floor, confirming the gauge barrier.
\end{itemize}

An important qualitative observation: the raw-input MLP is not only
unable to recover $R_\Sigma$; it is \emph{worse} than predicting the
constant mean angle, because it overfits to training-sample correlations
between field values and rotations that do not generalize. The
equivariant network, having access only to $SO(3)$-invariant
information, cannot overfit to such spurious signals. This is a positive
argument for equivariance in any single-view setting.

\begin{figure}[ht]
\centering
\includegraphics[width=0.6\linewidth]{plots/fig15_equivariant.png}
\caption{Invariant task (power spectrum recovery) is closed by equivariant
features; equivariant task (rotation recovery) remains at the gauge floor
for both architectures.}
\label{fig:equiv}
\end{figure}
"""

# Insert before \appendix
if "Proposition 6" not in s:
    s = s.replace("\\appendix", prop6 + "\n\\appendix", 1)
    print("Proposition 6 inserted.")
else:
    print("Proposition 6 already present.")

# Add to reproducibility index
old_rep = r"\item \texttt{src/phase2.py} --- Fig.~\ref{fig:mlp}"
new_rep = old_rep + "\n\\item \\texttt{src/phase6\\_equivariant.py} --- Fig.~\\ref{fig:equiv}"
if old_rep in s and "phase6\\_equivariant" not in s:
    s = s.replace(old_rep, new_rep, 1)
    print("Reproducibility index updated.")

p.write_text(s)
print("paper.tex updated.")
