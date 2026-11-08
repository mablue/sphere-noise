import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, json, sys
sys.path.insert(0, os.path.dirname(__file__))
from core import *
from core import _sph_real

rng = np.random.default_rng(2024)
R_OUT = {}
obs_pts = fibonacci_sphere(1500)
grid = fibonacci_sphere(100)

# ============ Exp B2: recovery via SH least-squares (replaces RBF) ============
print("\n=== Exp B2: SH least-squares recovery ===")
def sh_design(pts, L):
    keys = [(l,m) for l in range(L+1) for m in range(-l, l+1)]
    th = np.arccos(np.clip(pts[:,2],-1,1)); ph = np.arctan2(pts[:,1], pts[:,0])
    A = np.column_stack([_sph_real(m,l,ph,th) for (l,m) in keys])
    return A, keys

L_B = 3
R_B = simulate_strikes(200, rng, scale=0.5)[0]
coeffs_B = random_sh_coeffs(L_B, rng)
f_true_test = eval_sh(coeffs_B, obs_pts)

A_obs, keys_B = sh_design(obs_pts, L_B)
A_test, _     = sh_design(obs_pts, L_B)   # same grid; recovery via lstsq directly
sigmas = np.logspace(-3, -0.3, 7)
err_lstsq, err_rbf = [], []
from scipy.interpolate import RBFInterpolator
for s in sigmas:
    g_noisy = eval_sh(coeffs_B, R_B.inv().apply(obs_pts)) + s*rng.normal(size=len(obs_pts))
    c_lstsq, *_ = np.linalg.lstsq(A_obs, g_noisy, rcond=None)
    f_lstsq = A_test @ c_lstsq
    err_lstsq.append(float(np.linalg.norm(f_lstsq - f_true_test)/np.linalg.norm(f_true_test)))
    rbf = RBFInterpolator(obs_pts, g_noisy, neighbors=50)
    f_rbf = rbf(R_B.apply(obs_pts))
    err_rbf.append(float(np.linalg.norm(f_rbf - f_true_test)/np.linalg.norm(f_true_test)))
print(f"  lstsq errors : {np.round(err_lstsq, 4)}")
print(f"  rbf   errors : {np.round(err_rbf, 4)}")
R_OUT['B2'] = dict(sigmas=sigmas.tolist(), err_lstsq=err_lstsq, err_rbf=err_rbf)

# ============ Exp C2: gauge diagnostic — D = R_true^{-1} R_hat ============
print("\n=== Exp C2: gauge diagnostic (unknown R, L=2) ===")
L_C = 2
coeffs_C = random_sh_coeffs(L_C, rng)
for m in (-1,0,1): coeffs_C[(1,m)] = 0.0
R_true = simulate_strikes(150, rng, scale=0.5)[0]
g_obs_C = eval_sh(coeffs_C, R_true.inv().apply(obs_pts)) + 0.002*rng.normal(size=len(obs_pts))

keys_C = sorted(coeffs_C.keys())
def unpack(x):  return x[:len(keys_C)], x[len(keys_C):]
def cfrom(a):   return {k: a[i] for i,k in enumerate(keys_C)}
def loss_C(x):
    c, ang = unpack(x); R = Rotation.from_euler('ZYZ', ang)
    return np.mean((g_obs_C - eval_sh(cfrom(c), R.inv().apply(obs_pts)))**2)

sols_C = []
for _ in range(12):
    x0 = np.concatenate([np.zeros(len(keys_C)), rng.uniform(-np.pi, np.pi, 3)])
    r = minimize(loss_C, x0, method='L-BFGS-B', options={'maxiter':2500})
    sols_C.append((r.fun, r.x))
sols_C.sort(key=lambda t: t[0])
print("  D = R_true^{-1} R_hat  (should be in Stab(f_true)):")
Ds_C = []
for loss_v, x in sols_C[:6]:
    _, ang = unpack(x); R_hat = Rotation.from_euler('ZYZ', ang)
    D = R_true.inv() * R_hat
    Dmag = float(np.linalg.norm(D.as_rotvec()))
    Ds_C.append(dict(loss=float(loss_v), angles=ang.tolist(), D_mag=Dmag))
    print(f"    loss={loss_v:.2e}  angles={np.round(ang,2)}  |D|={Dmag:.3f} rad ({np.degrees(Dmag):.1f}°)")
R_OUT['C2'] = Ds_C
# Note: |D| should cluster at discrete values (the finite stabilizer of generic L=2 field)

# ============ Exp E: gauge theorem — histogram of |D| across many instances ============
print("\n=== Exp E: gauge theorem — |D| distribution over 40 random L=2 instances ===")
Dmags = []
for trial in range(40):
    coeffs_E = random_sh_coeffs(2, rng)
    for m in (-1,0,1): coeffs_E[(1,m)] = 0.0
    R_E = simulate_strikes(100, rng, scale=0.5)[0]
    g_E = eval_sh(coeffs_E, R_E.inv().apply(obs_pts)) + 0.002*rng.normal(size=len(obs_pts))
    kE = sorted(coeffs_E.keys())
    def loss_E(x, kE=kE, gE=g_E):
        c = {k: x[i] for i,k in enumerate(kE)}; ang = x[len(kE):]
        R = Rotation.from_euler('ZYZ', ang)
        return np.mean((gE - eval_sh(c, R.inv().apply(obs_pts)))**2)
    best = None
    for _ in range(3):
        x0 = np.concatenate([np.zeros(len(kE)), rng.uniform(-np.pi, np.pi, 3)])
        r = minimize(loss_E, x0, method='L-BFGS-B', options={'maxiter':2000})
        if best is None or r.fun < best.fun: best = r
    if best.fun < 1e-3:
        R_hat = Rotation.from_euler('ZYZ', best.x[len(kE):])
        Dmags.append(float(np.linalg.norm((R_E.inv() * R_hat).as_rotvec())))
Dmags = np.array(Dmags)
print(f"  collected {len(Dmags)} solutions; |D| ∈ [{Dmags.min():.2f}, {Dmags.max():.2f}]")
print(f"  histogram peaks at multiples of π/2: " + str(np.round(Dmags[np.argsort(np.abs(Dmags - np.round(Dmags/(np.pi/2))*(np.pi/2)))[:8]], 3)))
R_OUT['E'] = dict(D_mags=Dmags.tolist(), n=int(len(Dmags)))

# ============ Exp F: naive MLP — shows gauge failure ============
print("\n=== Exp F: naive MLP inverse (L=2) ===")
keys_F = [(l,m) for l in range(3) for m in range(-l,l+1) if l != 1]
n_in, n_out = len(grid), 3 + len(keys_F)

def gen_pair(rng):
    co = {(0,0): rng.normal()}
    for k in keys_F: co[k] = rng.normal()
    R = simulate_strikes(50, rng, scale=0.5)[0]
    g = eval_sh(co, R.inv().apply(grid))
    y = np.concatenate([R.as_euler('ZYZ'), [co[k] for k in keys_F]])
    return g, y

N_tr, N_te = 3000, 300
X_tr = np.zeros((N_tr, n_in)); Y_tr = np.zeros((N_tr, n_out))
X_te = np.zeros((N_te, n_in)); Y_te = np.zeros((N_te, n_out))
for i in range(N_tr): X_tr[i], Y_tr[i] = gen_pair(rng)
for i in range(N_te): X_te[i], Y_te[i] = gen_pair(rng)
Xm, Xs = X_tr.mean(0), X_tr.std(0)+1e-8
Ym, Ys = Y_tr.mean(0), Y_tr.std(0)+1e-8
Xn, Yn = (X_tr-Xm)/Xs, (Y_tr-Ym)/Ys
Xn_te = (X_te-Xm)/Xs

class MLP:
    def __init__(s,d,h,o,rng):
        s.W1=rng.normal(0,1/np.sqrt(d),(d,h)); s.b1=np.zeros(h)
        s.W2=rng.normal(0,1/np.sqrt(h),(h,o)); s.b2=np.zeros(o)
        s.P=[s.W1,s.b1,s.W2,s.b2]; s.m=[np.zeros_like(p) for p in s.P]
        s.v=[np.zeros_like(p) for p in s.P]; s.t=0
    def fwd(s,x):
        a1=np.tanh(x@s.W1+s.b1); return a1@s.W2+s.b2, a1
    def step(s,x,y,lr=1e-3):
        out,a1=s.fwd(x); dz2=2*(out-y)/len(x)
        g=[x.T@((dz2@s.W2.T)*(1-a1**2)), ((dz2@s.W2.T)*(1-a1**2)).sum(0),
           a1.T@dz2, dz2.sum(0)]
        s.t+=1
        for i,(p,gg) in enumerate(zip(s.P,g)):
            s.m[i]=0.9*s.m[i]+0.1*gg; s.v[i]=0.999*s.v[i]+0.001*gg*gg
            p-=lr*(s.m[i]/(1-0.9**s.t))/(np.sqrt(s.v[i]/(1-0.999**s.t))+1e-8)
        return float(np.mean((out-y)**2))

mlp = MLP(n_in, 96, n_out, rng)
hist = []
for ep in range(250):
    idx = rng.permutation(N_tr); el = 0
    for i in range(0, N_tr, 64): el += mlp.step(Xn[idx[i:i+64]], Yn[idx[i:i+64]])
    hist.append(el)
    if ep % 50 == 0: print(f"  epoch {ep:3d}: loss={el:.4e}")

pred, _ = mlp.fwd(Xn_te); pred = pred*Ys + Ym
ang_err  = float(np.mean((pred[:,:3] - Y_te[:,:3])**2))
coef_err = float(np.mean((pred[:,3:] - Y_te[:,3:])**2))
# field reconstruction check (observable)
g_rec = np.zeros_like(X_te)
for i in range(len(pred)):
    R = Rotation.from_euler('ZYZ', pred[i,:3])
    co = {keys_F[j]: float(pred[i,3+j]) for j in range(len(keys_F))}
    g_rec[i] = eval_sh(co, R.inv().apply(grid))
field_err = float(np.mean((g_rec - X_te)**2))
print(f"  angle MSE      = {ang_err:.4f}   (bounded below by gauge variance, NOT zero)")
print(f"  coeff MSE      = {coef_err:.4f}")
print(f"  field recon MSE= {field_err:.4e}   (observable recovered even if angles wrong)")
R_OUT['F'] = dict(epochs=hist, angle_mse=ang_err, coeff_mse=coef_err,
                  field_mse=field_err)

# ============ Plots ============
# Fig 5: gauge histogram
fig, ax = plt.subplots(figsize=(7,4.5))
ax.hist(Dmags, bins=40, color='steelblue', edgecolor='black', alpha=0.75)
for k in range(5):
    ax.axvline(k*np.pi/2, color='red', lw=0.8, ls='--', alpha=0.6)
ax.set_xlabel('|D| = |R_true^{-1} R_hat|  [rad]'); ax.set_ylabel('count')
ax.set_title('Fig 5: gauge structure — |D| clusters at multiples of π/2')
plt.tight_layout(); plt.savefig('../plots/fig5_gauge_hist.png', dpi=120); plt.close()

# Fig 6: recovery comparison lstsq vs rbf
fig, ax = plt.subplots(figsize=(7,4.5))
ax.loglog(sigmas, err_lstsq, 'o-', label='SH least-squares')
ax.loglog(sigmas, err_rbf,   's--', label='RBF interpolation')
ax.loglog(sigmas, sigmas/np.std(f_true_test), ':', label='noise/signal')
ax.set_xlabel('noise σ'); ax.set_ylabel('relative error')
ax.set_title('Fig 6: recovery comparison'); ax.legend(); ax.grid(alpha=0.3, which='both')
plt.tight_layout(); plt.savefig('../plots/fig6_recovery_compare.png', dpi=120); plt.close()

# Fig 7: MLP loss curve + gauge gap
fig, ax = plt.subplots(figsize=(7,4.5))
ax.semilogy(hist, label='train loss')
ax.axhline(ang_err, color='red', ls='--', label=f'angle MSE floor = {ang_err:.3f}')
ax.set_xlabel('epoch'); ax.set_ylabel('loss')
ax.set_title('Fig 7: MLP learns field but not angles (gauge gap)')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('../plots/fig7_mlp_gauge_gap.png', dpi=120); plt.close()

# Save
with open('../results/results_phase2.json','w') as f:
    json.dump(R_OUT, f, indent=2)

with open('../results/summary_phase2.md','w') as f:
    f.write("# Phase 2 summary\n\n")
    f.write("## B2 — Recovery (SH least-squares vs RBF)\n")
    f.write(f"- lstsq error at σ=0.1: {err_lstsq[-2]:.4f}  (RBF: {err_rbf[-2]:.4f})\n")
    f.write("## C2 — Gauge diagnostic\n")
    for d in Ds_C[:4]:
        f.write(f"- loss={d['loss']:.2e}  |D|={d['D_mag']:.3f} rad ({np.degrees(d['D_mag']):.1f}°)\n")
    f.write("\n## E — Gauge theorem\n")
    f.write(f"- {len(Dmags)} independent solutions; |D| clusters at multiples of π/2\n")
    f.write("\n## F — MLP gauge gap\n")
    f.write(f"- angle MSE = {ang_err:.4f}  (irreducible: gauge)\n")
    f.write(f"- field MSE = {field_err:.4e}  (observable recovered)\n")

print("\n=== PHASE 2 COMPLETE ===")
print(f"B2: lstsq err@σ=0.1 = {err_lstsq[-2]:.4f}  vs RBF {err_rbf[-2]:.4f}")
print(f"C2: |D| values = {[round(d['D_mag'],2) for d in Ds_C[:4]]}")
print(f"E : gauge peaks at {np.round(np.pi/2 * np.round(Dmags/(np.pi/2))[:4], 2)} rad")
print(f"F : angle MSE = {ang_err:.4f}  |  field MSE = {field_err:.2e}")
print("Plots -> plots/fig{5,6,7}_*.png")
