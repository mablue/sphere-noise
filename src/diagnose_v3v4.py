import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import differential_evolution
import sys; sys.path.insert(0, '.')
from core import *
from core import _sph_real

def sh_design(pts, L):
    keys = [(l,m) for l in range(L+1) for m in range(-l,l+1)]
    th = np.arccos(np.clip(pts[:,2],-1,1)); ph = np.arctan2(pts[:,1],pts[:,0])
    return np.column_stack([_sph_real(m,l,ph,th) for (l,m) in keys]), keys

obs_pts = fibonacci_sphere(1000)
Lb = 3
vk = [(l,m) for l in range(Lb+1) for m in range(-l,l+1)]

def run_v4(rng, label):
    print(f"\n--- {label} ---")
    Rb = simulate_strikes(200, rng, scale=0.5)[0]
    cb = random_sh_coeffs(Lb, rng)
    v  = np.array([cb[k] for k in vk])
    A_rot,_ = sh_design(Rb.inv().apply(obs_pts), Lb)
    A_ref,_ = sh_design(obs_pts, Lb)
    f_true  = A_ref @ v
    S = np.linalg.svd(A_rot, compute_uv=False)
    print(f"  ||v||={np.linalg.norm(v):.3f}, ||f_true||={np.linalg.norm(f_true):.3f}")
    print(f"  cond(A_rot)={S[0]/S[-1]:.2e}, S=({S[0]:.2f},...,{S[-1]:.4f})")
    print(f"  det(Rb)={np.linalg.det(Rb.as_matrix()):.6f}, ||RbRbT-I||={np.linalg.norm(Rb.as_matrix()@Rb.as_matrix().T-np.eye(3)):.2e}")
    rp = Rb.inv().apply(obs_pts)
    print(f"  ||rp||−1: min={np.abs(np.linalg.norm(rp,axis=1)-1).min():.2e}, max={np.abs(np.linalg.norm(rp,axis=1)-1).max():.2e}")
    for s in [0.001, 0.1, 0.5]:
        g = A_rot @ v + s*rng.normal(1000)
        c_hat, *_ = np.linalg.lstsq(A_rot, g, rcond=None)
        fh = A_ref @ c_hat
        err = np.linalg.norm(fh-f_true)/np.linalg.norm(f_true)
        print(f"  σ={s}: err={err:.3e}")

# ==== TEST 1: fresh rng like phase3 ====
run_v4(np.random.default_rng(7), "V4 fresh rng (mimics phase3 that WORKED)")

# ==== TEST 2: rng state after V1-V3 consumption ====
rng = np.random.default_rng(11)
_ = random_sh_coeffs(2, rng); _ = simulate_strikes(100, rng, scale=0.5)
for _ in range(4): _ = Rotation.random(random_state=rng)
_ = simulate_strikes(100, rng, scale=0.5); _ = rng.normal(1000)
for _ in range(20): _ = rng.uniform(-np.pi, np.pi, 3)
run_v4(rng, "V4 after V1-V3 rng use (PRODUCED BUG)")

# ==== TEST 3: V3 fix — rotvec + differential evolution ====
print("\n" + "="*60)
print("V3 FIX — Stab(Y_2^0)")
print("="*60)
rng = np.random.default_rng(42)
coeffs_z = {(l,m): 0.0 for l in range(3) for m in range(-l,l+1)}
coeffs_z[(2,0)] = 1.0
R_z = simulate_strikes(100, rng, scale=0.5)[0]
g_z = eval_sh(coeffs_z, R_z.inv().apply(obs_pts)) + 0.0005*rng.normal(1000)

def loss_rv(rv):
    Rh = Rotation.from_rotvec(rv)
    return float(np.mean((g_z - eval_sh(coeffs_z, Rh.inv().apply(obs_pts)))**2))

print(f"  loss at R_z itself: {loss_rv(R_z.as_rotvec()):.2e}   (should be ~1e-7)")
print(f"  loss at identity  : {loss_rv(np.zeros(3)):.2e}")

result = differential_evolution(loss_rv, [(-np.pi,np.pi)]*3,
                                seed=42, maxiter=400, tol=1e-11, polish=True)
print(f"  DE: loss={result.fun:.2e}")
if result.fun < 1e-4:
    R_hat = Rotation.from_rotvec(result.x)
    D = R_z.inv() * R_hat
    mag = np.linalg.norm(D.as_rotvec())
    axis = D.as_rotvec()/mag if mag > 1e-6 else np.zeros(3)
    cls = "SO(2)_z" if abs(abs(axis[2])-1)<0.1 else ("π_xy" if abs(axis[2])<0.1 and abs(mag-np.pi)<0.1 else "other")
    print(f"  |D|={mag:.3f}, axis={axis.round(3)}, class={cls}")

# ==== TEST 4: V3 with L-BFGS-B from R_z-init (sanity) ====
from scipy.optimize import minimize
res = minimize(loss_rv, R_z.as_rotvec(), method='L-BFGS-B',
               options={'maxiter':5000})
print(f"  L-BFGS-B from R_z.as_rotvec(): loss={res.fun:.2e}")

# Check all 10 random inits (as in original V3) — report ALL losses
print("\n  Diagnostics on original V3 approach (ZYZ + 20 random inits):")
def loss_zyz(ang):
    Rh = Rotation.from_euler('ZYZ', ang)
    return float(np.mean((g_z - eval_sh(coeffs_z, Rh.inv().apply(obs_pts)))**2))
losses = []
for i in range(20):
    x0 = rng.uniform(-np.pi, np.pi, 3)
    r = minimize(loss_zyz, x0, method='L-BFGS-B', options={'maxiter':4000})
    losses.append(r.fun)
losses = np.array(losses)
print(f"    min={losses.min():.2e}, median={np.median(losses):.2e}, max={losses.max():.2e}")
print(f"    count(loss < 1e-4) = {(losses < 1e-4).sum()}/20")
