import numpy as np
from scipy.spatial.transform import Rotation
import sys; sys.path.insert(0, '.')
from core import *
from core import _sph_real
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, os

rng = np.random.default_rng(11)

def sh_design(pts, L):
    keys = [(l,m) for l in range(L+1) for m in range(-l,l+1)]
    th = np.arccos(np.clip(pts[:,2],-1,1)); ph = np.arctan2(pts[:,1],pts[:,0])
    return np.column_stack([_sph_real(m,l,ph,th) for (l,m) in keys]), keys

print("\n=== V1: rank & orthonormality ===")
pts = fibonacci_sphere(4000)
w   = 4*np.pi/len(pts)          # quadrature weight on sphere
for L in (0,1,2,3):
    A,_ = sh_design(pts, L)
    rank = np.linalg.matrix_rank(A)
    gram = w * A.T @ A
    orth_err = float(np.linalg.norm(gram - np.eye(gram.shape[0])))
    cond = float(np.linalg.cond(A))
    print(f"  L={L}: rank={rank:2d}/{(L+1)**2:2d}   cond={cond:8.3f}   ||Gram-I||={orth_err:.2e}")

print("\n=== V2: SO(3)-torsor (rank-9/9 basis) ===")
L3 = 2
obs_pts = fibonacci_sphere(1000)
coeffs = random_sh_coeffs(L3, rng)
for m in (-1,0,1): coeffs[(1,m)] = 0.0
vec = np.array([coeffs[k] for k in [(l,m) for l in range(L3+1) for m in range(-l,l+1)]])

R_true = simulate_strikes(100, rng, scale=0.5)[0]

fit_idx  = rng.choice(len(obs_pts), 500, replace=False)
test_idx = np.setdiff1d(np.arange(len(obs_pts)), fit_idx)

print("  R_hat              fit_res   cv_res     ||c||")
for name, R_hat in [("CORRECT R_true", R_true),
                    ("identity", Rotation.identity())] + \
                   [(f"random #{i}", Rotation.random(random_state=rng)) for i in range(4)]:
    x_fit, x_te = obs_pts[fit_idx], obs_pts[test_idx]
    g_fit  = eval_sh(coeffs, R_true.inv().apply(x_fit))
    g_test = eval_sh(coeffs, R_true.inv().apply(x_te))

    A_fit, _ = sh_design(R_hat.inv().apply(x_fit), L3)
    A_te,  _ = sh_design(R_hat.inv().apply(x_te),  L3)
    c_hat, *_ = np.linalg.lstsq(A_fit, g_fit, rcond=None)
    fit_res = np.linalg.norm(A_fit@c_hat - g_fit)/np.linalg.norm(g_fit)
    f_pred  = A_te @ c_hat
    cv_res  = np.mean((f_pred - g_test)**2) / np.mean(g_test**2)
    print(f"  {name:<16} {fit_res:.2e}   {cv_res:.2e}   {np.linalg.norm(c_hat):.3f}")

print("\n  → اگر rank=9/9 درست باشد: CV residual برای همه ~1e-15 (torsor تأیید)")

print("\n=== V3: Stab(Y_2^0) — مجموعهٔ دقیق ===")
from scipy.optimize import minimize
coeffs_z = {(l,m): 0.0 for l in range(3) for m in range(-l,l+1)}
coeffs_z[(2,0)] = 1.0
R_z = simulate_strikes(100, rng, scale=0.5)[0]
g_z = eval_sh(coeffs_z, R_z.inv().apply(obs_pts)) + 0.0005*rng.normal(len(obs_pts))

def loss_Ronly(ang):
    Rh = Rotation.from_euler('ZYZ', ang)
    return float(np.mean((g_z - eval_sh(coeffs_z, Rh.inv().apply(obs_pts)))**2))

sols = []
for _ in range(20):
    x0 = rng.uniform(-np.pi, np.pi, 3)
    r  = minimize(loss_Ronly, x0, method='L-BFGS-B', options={'maxiter':4000})
    if r.fun < 1e-5:
        D = R_z.inv() * Rotation.from_euler('ZYZ', r.x)
        rv = D.as_rotvec(); mag = np.linalg.norm(rv)
        axis = rv/mag if mag > 1e-4 else np.zeros(3)
        sols.append(dict(loss=float(r.fun), Dmag=float(mag), axis=axis.tolist(),
                         axis_z=float(axis[2])))

print(f"  {len(sols)} sols found:")
for s in sols:
    cls = "SO(2)_z" if abs(abs(s['axis_z'])-1)<0.1 else ("π_xy" if abs(s['axis_z'])<0.1 else "other")
    print(f"    |D|={s['Dmag']:.3f}  axis={np.round(s['axis'],2)}  class={cls}")

print("\n  → Stab(Y_2^0) = SO(2)_z ∪ {R_π(n̂) : n̂ ⊥ ẑ}")

print("\n=== V4: B3 recovery with corrected basis ===")
Lb = 3; Rb = simulate_strikes(200, rng, scale=0.5)[0]
cb = random_sh_coeffs(Lb, rng)
vk = [(l,m) for l in range(Lb+1) for m in range(-l,l+1)]
v  = np.array([cb[k] for k in vk])
A_rot,_ = sh_design(Rb.inv().apply(obs_pts), Lb)
A_ref,_ = sh_design(obs_pts, Lb)
f_true = A_ref @ v

sigmas = np.logspace(-3,-0.3,7)
errs = []
for s in sigmas:
    g = A_rot @ v + s*rng.normal(len(obs_pts))
    c_hat, *_ = np.linalg.lstsq(A_rot, g, rcond=None)
    fh = A_ref @ c_hat
    errs.append(float(np.linalg.norm(fh-f_true)/np.linalg.norm(f_true)))
slope = float(np.polyfit(np.log(sigmas), np.log(errs), 1)[0])
print(f"  errors = {np.round(errs,4)}   slope = {slope:.2f}")

os.makedirs('../results', exist_ok=True)
with open('../results/verify.json','w') as f:
    json.dump(dict(sigmas=sigmas.tolist(), errs=errs, slope=slope,
                   sols=sols), f, indent=2)

# Fig
fig, ax = plt.subplots(figsize=(7,4.5))
ax.loglog(sigmas, errs, 'o-', label=f'SH lstsq (slope={slope:.2f})')
ax.loglog(sigmas, sigmas/np.std(f_true), ':', label='noise/signal')
ax.set_xlabel('σ'); ax.set_ylabel('rel error'); ax.legend(); ax.grid(alpha=0.3, which='both')
ax.set_title('Fig 11: recovery with corrected SH basis')
plt.tight_layout(); plt.savefig('../plots/fig11_recovery_final.png', dpi=120); plt.close()
print("\nOutputs → results/verify.json, plots/fig11_recovery_final.png")
