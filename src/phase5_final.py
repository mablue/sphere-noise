import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
import sys, json; sys.path.insert(0, '.')
from core import *
from core import _sph_real
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

rng = np.random.default_rng(2025)
obs_pts = fibonacci_sphere(1000)

def sh_design(pts, L):
    keys = [(l,m) for l in range(L+1) for m in range(-l,l+1)]
    th = np.arccos(np.clip(pts[:,2],-1,1)); ph = np.arctan2(pts[:,1],pts[:,0])
    return np.column_stack([_sph_real(m,l,ph,th) for (l,m) in keys]), keys

def classify_Delta(Delta, tag_axis, tol=0.1):
    """tag_axis = R2·ẑ : axis that SO(2)_z branch must be parallel to."""
    rv = Delta.as_rotvec(); mag = np.linalg.norm(rv)
    if mag < 1e-6: return "identity"
    ax = rv/mag
    cos_par = abs(np.dot(ax, tag_axis))
    if cos_par > 1 - tol: return "SO(2)"
    if abs(cos_par) < tol and abs(mag - np.pi) < tol: return "π_xy"
    return f"outside cos={cos_par:.2f},|D|={mag:.2f}"

# ============ Test A: generic f → Stab={I} → unique recovery ============
print("=== Test A: generic f → expect UNIQUE D  (err ≪ 1) ===")
L = 3; keysL = [(l,m) for l in range(L+1) for m in range(-l,l+1)]
c = random_sh_coeffs(L, rng)
R1 = simulate_strikes(50, rng, scale=0.5)[0]
R2 = simulate_strikes(50, rng, scale=0.5)[0]
g2_pts = eval_sh(c, R2.inv().apply(obs_pts))

A_obs,_ = sh_design(obs_pts, L)
g1_coeffs_vec, *_ = np.linalg.lstsq(A_obs, eval_sh(c, R1.inv().apply(obs_pts)), rcond=None)
g1_coeffs = {k: g1_coeffs_vec[i] for i,k in enumerate(keysL)}

def lossA(ang):
    D = Rotation.from_euler('ZYZ', ang)
    return float(np.mean((eval_sh(g1_coeffs, D.inv().apply(obs_pts)) - g2_pts)**2))

best = min((minimize(lossA, rng.uniform(-np.pi,np.pi,3), method='L-BFGS-B',
                     options={'maxiter':6000}) for _ in range(15)),
           key=lambda r: r.fun)
D_hat_A = Rotation.from_euler('ZYZ', best.x)
D_true_A = R2 * R1.inv()
errA = np.linalg.norm(D_hat_A.as_matrix() - D_true_A.as_matrix())
print(f"  loss={best.fun:.2e}   ‖D̂−D_true‖={errA:.2e}   {'✓' if errA<1e-3 else '✗'}")

# ============ Test B: Y_2^0 → unique up to Stab(f) ============
print("\n=== Test B: f = Y_2^0 → expect UNIQUE Δ ∈ Stab(f) ===")
cz = {(l,m): 0.0 for l in range(3) for m in range(-l,l+1)}
cz[(2,0)] = 1.0
R1 = simulate_strikes(50, rng, scale=0.5)[0]
R2 = simulate_strikes(50, rng, scale=0.5)[0]
g2_pts_B = eval_sh(cz, R2.inv().apply(obs_pts))

A2,_ = sh_design(obs_pts, 2)
keys2 = [(l,m) for l in range(3) for m in range(-l,l+1)]
g1v_B, *_ = np.linalg.lstsq(A2, eval_sh(cz, R1.inv().apply(obs_pts)), rcond=None)
g1_coeffsB = {k: g1v_B[i] for i,k in enumerate(keys2)}

def lossB(ang):
    D = Rotation.from_euler('ZYZ', ang)
    return float(np.mean((eval_sh(g1_coeffsB, D.inv().apply(obs_pts)) - g2_pts_B)**2))

R2z_axis = R2.apply(np.array([0.,0.,1.]))       # the correct test-axis for conjugation

sols = []
for _ in range(40):
    r = minimize(lossB, rng.uniform(-np.pi,np.pi,3), method='L-BFGS-B',
                 options={'maxiter':6000})
    if r.fun < 1e-8:
        D_hat = Rotation.from_euler('ZYZ', r.x)
        Delta_c = D_hat * (R2 * R1.inv()).inv()       # = R2 Δ R2⁻¹
        Delta_true = R2.inv() * Delta_c * R2          # ← restore Δ
        cls = classify_Delta(Delta_true, R2z_axis)
        sols.append(dict(loss=float(r.fun),
                         Delta_mag=float(np.linalg.norm(Delta_true.as_rotvec())),
                         Delta_axis=(Delta_true.as_rotvec()/max(np.linalg.norm(Delta_true.as_rotvec()),1e-9)).round(3).tolist(),
                         cls=cls))

print(f"  {len(sols)} recovered; loss all ≤ 1e-8")
by = {}
for s in sols: by[s['cls']] = by.get(s['cls'],0)+1
print(f"  breakdown: {by}")
print(f"  → ALL Δ ∈ Stab(Y_2^0)?  {all(s['cls'] in ('SO(2)','π_xy','identity') for s in sols)}")
print(f"  → SO(2) angles (continuous): {sorted([round(s['Delta_mag'],2) for s in sols if s['cls']=='SO(2)'])}")
print(f"  → π_xy angles  (discrete≈π): {sorted([round(s['Delta_mag'],2) for s in sols if s['cls']=='π_xy'])}")

# ============ Fig 14 ============
fig, ax = plt.subplots(figsize=(7,4.5))
so2 = [s['Delta_mag'] for s in sols if s['cls']=='SO(2)']
pixy = [s['Delta_mag'] for s in sols if s['cls']=='π_xy']
if so2:  ax.scatter(so2,  [1.0]*len(so2),  c='steelblue', s=100, label='SO(2)_z branch (continuous)')
if pixy: ax.scatter(pixy, [0.0]*len(pixy), c='crimson',   s=100, label='π_xy branch (|Δ|=π)')
ax.axvline(np.pi, color='k', ls=':', alpha=0.4)
ax.set_yticks([0,1]); ax.set_yticklabels(['π_xy','SO(2)_z'])
ax.set_xlabel('|Δ_true| = |R_2⁻¹ D̂ D_true⁻¹ R_2|  [rad]')
ax.set_title('Fig 14 (corrected): Δ_true always lies in Stab(f)')
ax.legend(); ax.grid(alpha=0.3, axis='x')
plt.tight_layout(); plt.savefig('../plots/fig14_multiview_stab.png', dpi=140); plt.close()

with open('../results/phase5_final.json','w') as f:
    json.dump(dict(A=dict(loss=float(best.fun), err=float(errA)),
                   B=dict(sols=sols, R2z=R2z_axis.tolist())), f, indent=2)
print("\nSaved results/phase5_final.json, plots/fig14_multiview_stab.png")
