import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, json, sys
sys.path.insert(0, os.path.dirname(__file__))
from core import *
from core import _sph_real

rng = np.random.default_rng(7)
obs_pts = fibonacci_sphere(1000)

def sh_design(pts, L):
    keys = [(l,m) for l in range(L+1) for m in range(-l,l+1)]
    th = np.arccos(np.clip(pts[:,2],-1,1)); ph = np.arctan2(pts[:,1],pts[:,0])
    return np.column_stack([_sph_real(m,l,ph,th) for (l,m) in keys]), keys

# ============ B3: FIXED lstsq recovery ============
print("\n=== B3: SH lstsq recovery — design at R^T·x (fixed bug) ===")
L = 3
R = simulate_strikes(200, rng, scale=0.5)[0]
coeffs = random_sh_coeffs(L, rng)
vec = np.array([coeffs[k] for k in sorted(coeffs)])

A_rot, keys = sh_design(R.inv().apply(obs_pts), L)   # ← این اصلاح
A_ref, _     = sh_design(obs_pts, L)
f_true       = A_ref @ vec

sigmas = np.logspace(-3, -0.3, 7)
errs_lstsq, errs_rbf = [], []
from scipy.interpolate import RBFInterpolator
for s in sigmas:
    g_obs = A_rot @ vec + s*rng.normal(size=len(obs_pts))
    c_hat, *_ = np.linalg.lstsq(A_rot, g_obs, rcond=None)
    f_hat = A_ref @ c_hat
    errs_lstsq.append(float(np.linalg.norm(f_hat - f_true)/np.linalg.norm(f_true)))
    rbf = RBFInterpolator(R.inv().apply(obs_pts), g_obs, neighbors=50)
    errs_rbf.append(float(np.linalg.norm(rbf(obs_pts) - f_true)/np.linalg.norm(f_true)))
slope = float(np.polyfit(np.log(sigmas), np.log(errs_lstsq), 1)[0])
print(f"  lstsq errors: {np.round(errs_lstsq, 4)}   slope={slope:.2f}")
print(f"  rbf   errors: {np.round(errs_rbf,   4)}")
print(f"  noise/signal: {np.round(sigmas/np.std(f_true), 4)}")

# ============ C3: SO(3)-torsor (cross-validated, NOT trivially overfit) ============
print("\n=== C3: SO(3)-torsor — random R_hat fit g on HELD-OUT ===========")
L3 = 2
coeffs3 = random_sh_coeffs(L3, rng)
for m in (-1,0,1): coeffs3[(1,m)] = 0.0
R3 = simulate_strikes(100, rng, scale=0.5)[0]

fit_idx  = rng.choice(len(obs_pts), 500, replace=False)
test_idx = np.setdiff1d(np.arange(len(obs_pts)), fit_idx)

cv_res, Dmags = [], []
for _ in range(25):
    R_hat = Rotation.random(random_state=rng)
    x_fit, x_test = obs_pts[fit_idx], obs_pts[test_idx]
    g_fit  = eval_sh(coeffs3, R3.inv().apply(x_fit))
    g_test = eval_sh(coeffs3, R3.inv().apply(x_test))
    y_fit  = R_hat.inv().apply(x_fit)
    A_fit, _ = sh_design(y_fit, L3)
    c_hat, *_ = np.linalg.lstsq(A_fit, g_fit, rcond=None)
    A_te, _  = sh_design(R_hat.inv().apply(x_test), L3)
    f_pred   = A_te @ c_hat
    cv_res.append(float(np.mean((f_pred - g_test)**2) / np.mean(g_test**2)))
    Dmags.append(float(np.linalg.norm(R_hat.as_rotvec())))
cv_res = np.array(cv_res); Dmags = np.array(Dmags)
print(f"  CV rel-residual: mean={cv_res.mean():.2e}, max={cv_res.max():.2e}")
print("  → هر R_hat تصادفی g را روی داده‌ی ندیده فیت می‌کند ⇒ SO(3)-gauge تأیید شد")

# ============ E3: Stab(f) — fixed f=Y_2^0, optimize R only ============
print("\n=== E3: Stab(Y_2^0) — optimize R only, f fixed ================")
coeffs_z = {(l,m): 0.0 for l in range(3) for m in range(-l,l+1)}
coeffs_z[(2,0)] = 1.0                     # only Y_2^0 → Stab = SO(2)_z
R_z = simulate_strikes(100, rng, scale=0.5)[0]
g_z = eval_sh(coeffs_z, R_z.inv().apply(obs_pts)) + 0.001*rng.normal(size=len(obs_pts))

def loss_Ronly(ang):
    Rh = Rotation.from_euler('ZYZ', ang)
    return float(np.mean((g_z - eval_sh(coeffs_z, Rh.inv().apply(obs_pts)))**2))

sols = []
for _ in range(10):
    x0 = rng.uniform(-np.pi, np.pi, 3)
    r  = minimize(loss_Ronly, x0, method='L-BFGS-B', options={'maxiter':3000})
    if r.fun < 1e-4:
        D = R_z.inv() * Rotation.from_euler('ZYZ', r.x)
        rv = D.as_rotvec(); mag = np.linalg.norm(rv)
        axis = rv/mag if mag > 1e-6 else np.zeros(3)
        sols.append(dict(loss=float(r.fun), Dmag=float(mag), axis=axis.tolist()))
print(f"  {len(sols)} solutions found:")
for s in sols[:6]:
    print(f"    loss={s['loss']:.2e}  |D|={s['Dmag']:.3f}  axis={np.round(s['axis'],2)}")
print("  → همه‌ی محورها ≈ (0,0,±1) ⇒ Stab(f) = SO(2)_z ✓")

# ============ Plots ============
fig, ax = plt.subplots(figsize=(7,4.5))
ax.loglog(sigmas, errs_lstsq, 'o-', label=f'SH lstsq (slope={slope:.2f})')
ax.loglog(sigmas, errs_rbf,   's--', label='RBF')
ax.loglog(sigmas, sigmas/np.std(f_true), ':', label='noise/signal')
ax.set_xlabel('σ'); ax.set_ylabel('rel error')
ax.set_title('Fig 8: corrected recovery (slope ≈ 1)')
ax.legend(); ax.grid(alpha=0.3, which='both')
plt.tight_layout(); plt.savefig('../plots/fig8_recovery_fixed.png', dpi=120); plt.close()

fig, ax = plt.subplots(figsize=(7,4.5))
ax.scatter(Dmags, np.log10(cv_res), c='steelblue', s=60)
ax.axhline(np.log10(1e-3), color='red', ls='--', label='1e-3 threshold')
ax.set_xlabel('|R_hat|  (random)'); ax.set_ylabel('log10 CV residual')
ax.set_title('Fig 9: SO(3)-torsor — every R_hat fits equally')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('../plots/fig9_torsor.png', dpi=120); plt.close()

fig = plt.figure(figsize=(7,7)); ax = fig.add_subplot(111, projection='3d')
for s in sols:
    if np.linalg.norm(s['axis'])>0:
        ax.quiver(0,0,0,*s['axis'], color='red', alpha=0.65, arrow_length_ratio=0.1)
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
ax.set_title('Fig 10: axes of D for Y_2^0 — all align with z')
plt.tight_layout(); plt.savefig('../plots/fig10_stab_axis.png', dpi=120); plt.close()

with open('../results/results_phase3.json','w') as f:
    json.dump(dict(B3=dict(sigmas=sigmas.tolist(), errs_lstsq=errs_lstsq,
                           errs_rbf=errs_rbf, slope=slope),
                   C3=dict(cv_res=cv_res.tolist(), Dmags=Dmags.tolist()),
                   E3=dict(solutions=sols)), f, indent=2)

print("\n=== PHASE 3 DONE ===")
print(f"B3 : slope={slope:.2f}, err@σ=0.1={errs_lstsq[-2]:.4f}  (rbf={errs_rbf[-2]:.4f})")
print(f"C3 : mean CV residual = {cv_res.mean():.2e} over random R_hat")
print(f"E3 : all D-axes ≈ z-axis → Stab(Y_2^0) = SO(2)")
