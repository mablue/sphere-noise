import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from scipy.interpolate import RBFInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, json, sys
sys.path.insert(0, os.path.dirname(__file__))
from core import *

rng = np.random.default_rng(42)
os.makedirs('../plots', exist_ok=True)
os.makedirs('../results', exist_ok=True)
R_OUT = {}

# ============ Exp A: SO(3) closure & Euler compression ============
print("\n=== Exp A: SO(3) closure & Euler compression ===")
N_A = 500
R_total, Rs, strike_pts, omegas, dts = simulate_strikes(N_A, rng, scale=0.5)

M = R_total.as_matrix()
orth_err = float(np.linalg.norm(M @ M.T - np.eye(3)))
det = float(np.linalg.det(M))
euler = R_total.as_euler('ZYZ')
R_rec = Rotation.from_euler('ZYZ', euler)
euler_err = float(np.linalg.norm(M - R_rec.as_matrix()))

print(f"  ||R R^T - I||      = {orth_err:.2e}")
print(f"  det(R)             = {det:.10f}")
print(f"  Euler (ZYZ)        = {euler}")
print(f"  Euler recon. error = {euler_err:.2e}")
print(f"  Compression: {N_A} strikes -> 3 numbers  (ratio {3*N_A}/3 = {N_A}x)")

R_OUT['A'] = dict(N=N_A, orth_err=orth_err, det=det,
                  euler=euler.tolist(), euler_recon_err=euler_err,
                  compression_ratio=N_A)

# ============ Exp B: recovery with known R ============
print("\n=== Exp B: recovery with known R ===")
L_B = 3
M_pts = 2000
test_pts = fibonacci_sphere(M_pts)
obs_pts  = fibonacci_sphere(M_pts)
coeffs_B = random_sh_coeffs(L_B, rng)
f_true   = eval_sh(coeffs_B, test_pts)

noise_sigma = 0.1
# g(q) = f(R^T q) + eps
g_at_obs = eval_sh(coeffs_B, R_total.inv().apply(obs_pts)) + noise_sigma * rng.normal(size=M_pts)
rbf = RBFInterpolator(obs_pts, g_at_obs, neighbors=50)
recover_pts = R_total.apply(test_pts)   # R @ test
f_hat = rbf(recover_pts)
err_B = float(np.linalg.norm(f_hat - f_true)/np.linalg.norm(f_true))
noise_rel = float(noise_sigma/np.std(f_true))
print(f"  Rel. recovery error = {err_B:.4f}")
print(f"  Rel. noise level    = {noise_rel:.4f}")

R_OUT['B'] = dict(L=L_B, M=M_pts, noise_sigma=noise_sigma,
                  rel_err=err_B, rel_noise=noise_rel)

# ============ Exp C: recovery with UNKNOWN R (diffusion style) ============
print("\n=== Exp C: recovery with unknown R ===")
L_C = 2
M_obs = 1500
obs_pts_C = fibonacci_sphere(M_obs)
coeffs_C = random_sh_coeffs(L_C, rng)
for m in (-1,0,1):                       # remove dipole (continuous gauge)
    coeffs_C[(1,m)] = 0.0
g_obs = eval_sh(coeffs_C, R_total.inv().apply(obs_pts_C)) + 0.01*rng.normal(size=M_obs)

keys = [(l,m) for l in range(L_C+1) for m in range(-l, l+1)]
def unpack(x):
    a = x[:len(keys)]
    return {k: a[i] for i,k in enumerate(keys)}, x[len(keys):]
def loss(x):
    coeffs, angles = unpack(x)
    R = Rotation.from_euler('ZYZ', angles)
    fv = eval_sh(coeffs, R.inv().apply(obs_pts_C))
    return float(np.mean((g_obs - fv)**2)) + 1e-6*np.sum(x[:len(keys)]**2)

best = None
for _ in range(6):
    x0 = np.concatenate([np.zeros(len(keys)), rng.uniform(-np.pi, np.pi, 3)])
    r = minimize(loss, x0, method='L-BFGS-B', options={'maxiter':3000})
    if best is None or r.fun < best.fun:
        best = r
coeffs_hat, angles_hat = unpack(best.x)
f_hat_C = eval_sh(coeffs_hat, test_pts)
err_C = float(np.linalg.norm(f_hat_C - f_true[:M_pts])/np.linalg.norm(f_true[:M_pts]))
print(f"  Final loss         = {best.fun:.4e}")
print(f"  Recovered angles   = {angles_hat}")
print(f"  True total angles  = {R_total.as_euler('ZYZ')}")

R_OUT['C'] = dict(L=L_C, M=M_obs, loss=float(best.fun),
                  recovered_angles=angles_hat.tolist(),
                  true_angles=R_total.as_euler('ZYZ').tolist())

# ============ Plots ============
print("\n=== Generating plots ===")

# Fig 1: strike distribution on S^2
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')
u = np.linspace(0, 2*np.pi, 30); v = np.linspace(0, np.pi, 20)
xs = np.outer(np.cos(u), np.sin(v)); ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones_like(u), np.cos(v))
ax.plot_wireframe(xs, ys, zs, color='gray', alpha=0.25, linewidth=0.4)
ax.scatter(strike_pts[:200,0], strike_pts[:200,1], strike_pts[:200,2],
           c='red', s=12, label='strike points')
ax.set_title(f'Fig 1: {N_A} strikes on S²')
ax.legend()
plt.savefig('../plots/fig1_strikes.png', dpi=120, bbox_inches='tight')
plt.close()

# Fig 2: original / observed / recovered
lon = np.arctan2(test_pts[:,1], test_pts[:,0])
lat = np.arcsin(np.clip(test_pts[:,2], -1, 1))
g_at_test = eval_sh(coeffs_B, R_total.inv().apply(test_pts))
fig, axes = plt.subplots(1,3, figsize=(15,4.5))
for ax, vals, title in zip(axes,
        [f_true, g_at_test, f_hat],
        ['original f', 'observed g = R·f', f'recovered f̂  (err={err_B:.3f})']):
    sc = ax.scatter(lon, lat, c=vals, cmap='RdBu_r', s=4)
    ax.set_title(title); ax.set_xlabel('lon'); ax.set_ylabel('lat')
    plt.colorbar(sc, ax=ax)
plt.tight_layout()
plt.savefig('../plots/fig2_recovery.png', dpi=120, bbox_inches='tight')
plt.close()

# Fig 3: error vs noise
print("  sweeping noise...")
sigmas = np.logspace(-3, -0.3, 8)
errs = []
for s in sigmas:
    g_n = eval_sh(coeffs_B, R_total.inv().apply(obs_pts)) + s*rng.normal(size=M_pts)
    rb = RBFInterpolator(obs_pts, g_n, neighbors=50)
    errs.append(float(np.linalg.norm(rb(recover_pts)-f_true)/np.linalg.norm(f_true)))
fig, ax = plt.subplots(figsize=(7,4.5))
ax.loglog(sigmas, errs, 'o-', label='recovery error')
ax.loglog(sigmas, sigmas/np.std(f_true), '--', label='noise / signal')
ax.set_xlabel('noise sigma'); ax.set_ylabel('relative error')
ax.set_title('Fig 3: recovery error vs noise')
ax.legend(); ax.grid(True, which='both', alpha=0.3)
plt.savefig('../plots/fig3_noise_sweep.png', dpi=120, bbox_inches='tight')
plt.close()
R_OUT['D_sweep'] = dict(sigmas=sigmas.tolist(), errs=errs)

# Fig 4: Euler reconstruction error vs N
Ns = [1, 5, 10, 50, 100, 500, 1000]
errsN = []
for n in Ns:
    Rn,_,_,_,_ = simulate_strikes(n, np.random.default_rng(7), scale=0.5)
    en = Rn.as_euler('ZYZ')
    errsN.append(max(float(np.linalg.norm(Rn.as_matrix() -
                              Rotation.from_euler('ZYZ', en).as_matrix())), 1e-16))
fig, ax = plt.subplots(figsize=(7,4.5))
ax.semilogy(Ns, errsN, 'o-')
ax.set_xlabel('N strikes'); ax.set_ylabel('Euler recon. error')
ax.set_title('Fig 4: 3 Euler numbers suffice for any N')
ax.grid(True, which='both', alpha=0.3)
plt.savefig('../plots/fig4_euler_scaling.png', dpi=120, bbox_inches='tight')
plt.close()

# ============ summary ============
with open('../results/results.json','w') as f:
    json.dump(R_OUT, f, indent=2)

with open('../results/summary.md','w') as f:
    f.write("# Run summary\n\n")
    f.write(f"- **A. Euler compression**: error = {euler_err:.2e}, ratio = {N_A}x\n")
    f.write(f"- **B. Known-R recovery**: err = {err_B:.4f}  (noise = {noise_rel:.4f})\n")
    f.write(f"- **C. Unknown-R recovery**: loss = {best.fun:.4e}\n\n")
    f.write("### Recovered vs true total rotation\n")
    f.write(f"- true     : {R_total.as_euler('ZYZ').round(3).tolist()}\n")
    f.write(f"- recovered: {np.round(angles_hat,3).tolist()}\n")

print("\n=== SUMMARY ===")
print(f"A: Euler compression error       = {euler_err:.2e}  ({N_A}x compression)")
print(f"B: Known-R recovery rel. error   = {err_B:.4f}  (noise {noise_rel:.4f})")
print(f"C: Unknown-R fit loss            = {best.fun:.4e}")
print("Results  -> results/results.json, results/summary.md")
print("Plots    -> plots/fig{1..4}_*.png")
