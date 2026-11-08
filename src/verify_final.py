import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize, differential_evolution
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os, json; sys.path.insert(0, '.')
from core import *
from core import _sph_real

rng = np.random.default_rng(11)
obs_pts = fibonacci_sphere(1000)

def sh_design(pts, L):
    keys = [(l,m) for l in range(L+1) for m in range(-l,l+1)]
    th = np.arccos(np.clip(pts[:,2],-1,1)); ph = np.arctan2(pts[:,1],pts[:,0])
    return np.column_stack([_sph_real(m,l,ph,th) for (l,m) in keys]), keys

out = {}

# ============ V1 — rank & orthonormality ============
print("\n=== V1: rank & orthonormality ===")
pts_quad = fibonacci_sphere(4000)
w = 4*np.pi/len(pts_quad)
V1 = {}
for L in (0,1,2,3):
    A,_ = sh_design(pts_quad, L)
    V1[L] = dict(rank=int(np.linalg.matrix_rank(A)),
                 cond=float(np.linalg.cond(A)),
                 gram_err=float(np.linalg.norm(w*A.T@A - np.eye((L+1)**2))))
    print(f"  L={L}: rank={V1[L]['rank']}/{(L+1)**2}  cond={V1[L]['cond']:.3f}  ||Gram-I||={V1[L]['gram_err']:.1e}")
out['V1'] = V1

# ============ V4-fixed — recovery with CORRECT noise generator ============
print("\n=== V4-fixed: recovery (noise = standard normal, size=N) ===")
Lb = 3
Rb = simulate_strikes(200, rng, scale=0.5)[0]
cb = random_sh_coeffs(Lb, rng)
vk = [(l,m) for l in range(Lb+1) for m in range(-l,l+1)]
v  = np.array([cb[k] for k in vk])
A_rot,_ = sh_design(Rb.inv().apply(obs_pts), Lb)
A_ref,_ = sh_design(obs_pts, Lb)
f_true  = A_ref @ v
print(f"  ||v||={np.linalg.norm(v):.3f}   ||f_true||={np.linalg.norm(f_true):.3f}   cond(A_rot)={np.linalg.cond(A_rot):.3f}")

sigmas = np.logspace(-3, -0.3, 7)
errs_clean = []
for s in sigmas:
    g = A_rot @ v + s * rng.standard_normal(len(obs_pts))   # ← درست
    c_hat, *_ = np.linalg.lstsq(A_rot, g, rcond=None)
    f_hat = A_ref @ c_hat
    errs_clean.append(float(np.linalg.norm(f_hat - f_true)/np.linalg.norm(f_true)))
print(f"  errors     = {np.round(errs_clean, 4)}")
print(f"  CRB (σ/||f||·√(d/N)·||f||) ~ {np.round(sigmas/np.std(f_true)*np.sqrt(16/1000), 4)}")
slope = float(np.polyfit(np.log(sigmas), np.log(errs_clean), 1)[0])
print(f"  slope = {slope:.3f}")
out['V4'] = dict(sigmas=sigmas.tolist(), errs=errs_clean, slope=slope)

# ============ V3-fixed — Stab(Y_2^0) with clean noise ============
print("\n=== V3-fixed: Stab(Y_2^0) — clean noise, rotvec optimizer ===")
coeffs_z = {(l,m): 0.0 for l in range(3) for m in range(-l,l+1)}
coeffs_z[(2,0)] = 1.0
R_z = simulate_strikes(100, rng, scale=0.5)[0]
g_z = eval_sh(coeffs_z, R_z.inv().apply(obs_pts)) + 0.001 * rng.standard_normal(len(obs_pts))

def loss_rv(rv):
    Rh = Rotation.from_rotvec(rv)
    return float(np.mean((g_z - eval_sh(coeffs_z, Rh.inv().apply(obs_pts)))**2))

print(f"  loss at R_z itself : {loss_rv(R_z.as_rotvec()):.2e}   (should be ~1e-6)")
print(f"  loss at identity   : {loss_rv(np.zeros(3)):.2e}")

sols = []
# multi-start L-BFGS-B from random rotvec + DE confirmation
for _ in range(20):
    r = minimize(loss_rv, rng.uniform(-np.pi, np.pi, 3),
                 method='L-BFGS-B', options={'maxiter':5000})
    if r.fun < 1e-4:
        D = R_z.inv() * Rotation.from_rotvec(r.x)
        mag = np.linalg.norm(D.as_rotvec())
        ax  = D.as_rotvec()/mag if mag > 1e-6 else np.zeros(3)
        sols.append(dict(loss=float(r.fun), Dmag=float(mag), axis=ax.tolist()))

print(f"  {len(sols)} solutions found:")
for s in sols:
    ax = np.array(s['axis']); mag = s['Dmag']
    cls = "SO(2)_z" if abs(abs(ax[2])-1) < 0.1 \
          else "π_xy (⊥z, |D|=π)" if abs(ax[2]) < 0.1 and abs(mag-np.pi) < 0.1 \
          else "other"
    print(f"    loss={s['loss']:.1e}  |D|={mag:.3f}  axis={np.round(ax,2)}  class={cls}")
out['V3'] = sols

# ============ V2 — torsor sanity (should already pass) ============
print("\n=== V2: SO(3)-torsor (sanity) ===")
L2 = 2
c2 = random_sh_coeffs(L2, rng)
for m in (-1,0,1): c2[(1,m)] = 0.0
R2 = simulate_strikes(100, rng, scale=0.5)[0]
g_t = eval_sh(c2, R2.inv().apply(obs_pts)) + 0.001 * rng.standard_normal(len(obs_pts))
vk2 = [(l,m) for l in range(L2+1) for m in range(-l,l+1)]
v2  = np.array([c2[k] for k in vk2])
res = []
for name, Rh in [("R_true", R2), ("I", Rotation.identity()),
                 ("random1", Rotation.random(random_state=rng)),
                 ("random2", Rotation.random(random_state=rng))]:
    A,_ = sh_design(Rh.inv().apply(obs_pts), L2)
    ch, *_ = np.linalg.lstsq(A, g_t, rcond=None)
    fit_res = float(np.linalg.norm(A@ch - g_t) / np.linalg.norm(g_t))
    res.append(dict(name=name, fit_res=fit_res, c_norm=float(np.linalg.norm(ch))))
    print(f"  {name:<8}: fit_res={fit_res:.2e}  ||c||={np.linalg.norm(ch):.3f}")
out['V2'] = res

# ============ Plot ============
fig, ax = plt.subplots(figsize=(7,4.5))
ax.loglog(sigmas, errs_clean, 'o-', label=f'SH lstsq (slope={slope:.2f})')
ax.loglog(sigmas, sigmas/np.std(f_true)*np.sqrt(16/1000), ':', label='theory √(d/N)·σ/‖f‖')
ax.set_xlabel('σ'); ax.set_ylabel('rel error'); ax.legend(); ax.grid(alpha=0.3, which='both')
ax.set_title('Fig 12: corrected recovery (slope ≈ 1, matches CRB)')
plt.tight_layout(); plt.savefig('../plots/fig12_recovery_correct.png', dpi=120); plt.close()

with open('../results/verify_final.json','w') as f:
    json.dump(out, f, indent=2)
print("\nOutputs → results/verify_final.json, plots/fig12_recovery_correct.png")
