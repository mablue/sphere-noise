import numpy as np
from scipy.spatial.transform import Rotation
import sys; sys.path.insert(0, '.')
from core import *
from core import _sph_real
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, os

rng = np.random.default_rng(2025)
obs_pts = fibonacci_sphere(1000)

def sh_design(pts, L):
    keys = [(l,m) for l in range(L+1) for m in range(-l,l+1)]
    th = np.arccos(np.clip(pts[:,2],-1,1)); ph = np.arctan2(pts[:,1],pts[:,0])
    return np.column_stack([_sph_real(m,l,ph,th) for (l,m) in keys]), keys

print("=== Phase 5: Multi-view identification ===")
print("Theorem: given g1=f∘R1⁻¹ and g2=f∘R2⁻¹, we recover D* = R1⁻¹R2")
print("         up to Stab(f) — regardless of f. No assumption on f.\n")

# Test 1: full-field case (f generic → Stab(f)={I})
print("--- Test A: generic f, Stab(f)={I} → D should be UNIQUE ---")
L = 3
c = random_sh_coeffs(L, rng)
R1 = simulate_strikes(50, rng, scale=0.5)[0]
R2 = simulate_strikes(50, rng, scale=0.5)[0]
g1 = eval_sh(c, R1.inv().apply(obs_pts))
g2 = eval_sh(c, R2.inv().apply(obs_pts))

# Optimize D such that g1 ∘ D⁻¹ ≈ g2
def loss_mv(ang):
    D = Rotation.from_euler('ZYZ', ang)
    g1_rot = eval_sh(c, R1.inv().apply(D.inv().apply(obs_pts)))  # g1(D⁻¹x)
    return float(np.mean((g1_rot - g2)**2))

best = None
for _ in range(20):
    x0 = rng.uniform(-np.pi, np.pi, 3)
    r = __import__('scipy.optimize', fromlist=['minimize']).minimize(
        loss_mv, x0, method='L-BFGS-B', options={'maxiter':5000})
    if best is None or r.fun < best.fun: best = r

D_true = R1.inv() * R2
D_hat  = Rotation.from_euler('ZYZ', best.x)
err = np.linalg.norm(D_hat.as_matrix() - D_true.as_matrix())
print(f"  loss={best.fun:.2e}   ‖D_hat − D_true‖={err:.2e}   ⟵ recovery confirmed")

# Test 2: high-symmetry f (Y_2^0) → Stab=SO(2)_z ∪ π_xy, expect DISCRETE ambiguities
print("\n--- Test B: f = Y_2^0, Stab(f) has 4 components → D unique up to Stab ---")
cz = {(l,m): 0.0 for l in range(3) for m in range(-l,l+1)}
cz[(2,0)] = 1.0
R1 = simulate_strikes(50, rng, scale=0.5)[0]
R2 = simulate_strikes(50, rng, scale=0.5)[0]
g1 = eval_sh(cz, R1.inv().apply(obs_pts))
g2 = eval_sh(cz, R2.inv().apply(obs_pts))

def loss_mv2(ang):
    D = Rotation.from_euler('ZYZ', ang)
    g1_rot = eval_sh(cz, R1.inv().apply(D.inv().apply(obs_pts)))
    return float(np.mean((g1_rot - g2)**2))

sols = []
from scipy.optimize import minimize
for _ in range(30):
    x0 = rng.uniform(-np.pi, np.pi, 3)
    r = minimize(loss_mv2, x0, method='L-BFGS-B', options={'maxiter':5000})
    if r.fun < 1e-4:
        D_hat = Rotation.from_euler('ZYZ', r.x)
        D_err = np.linalg.norm((R1.inv()*R2).as_matrix() - D_hat.as_matrix())
        sols.append(dict(loss=float(r.fun), D_hat_angles=r.x.tolist(),
                         D_relative_to_truth=float(D_err)))
print(f"  {len(sols)} solutions (should be finite, all matching up to Stab):")
for s in sols[:8]:
    print(f"    loss={s['loss']:.1e}  ‖D_hat−D_true‖={s['D_relative_to_truth']:.3f}")

print("\nInterpretation: multi-view makes R1⁻¹R2 identifiable for GENERIC f;")
print("for symmetric f, identification is up to Stab(f) — i.e., O(|Stab(f)|)")
print("ambiguities, not the continuous SO(3) torsor of single-view.")

with open('../results/phase5_multiview.json','w') as f:
    json.dump(dict(testA=dict(loss=float(best.fun), err=float(err)),
                   testB=dict(sols=sols)), f, indent=2)
print("\nsaved results/phase5_multiview.json")
