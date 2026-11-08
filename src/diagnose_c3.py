import numpy as np
from scipy.spatial.transform import Rotation
import sys; sys.path.insert(0, '.')
from core import *
from core import _sph_real

rng = np.random.default_rng(7)
obs_pts = fibonacci_sphere(1000)

def sh_design(pts, L):
    keys = [(l,m) for l in range(L+1) for m in range(-l,l+1)]
    th = np.arccos(np.clip(pts[:,2],-1,1)); ph = np.arctan2(pts[:,1],pts[:,0])
    return np.column_stack([_sph_real(m,l,ph,th) for (l,m) in keys]), keys

L3 = 2
coeffs3 = random_sh_coeffs(L3, rng)
for m in (-1,0,1): coeffs3[(1,m)] = 0.0
R3 = simulate_strikes(100, rng, scale=0.5)[0]

fit_idx  = rng.choice(len(obs_pts), 500, replace=False)
test_idx = np.setdiff1d(np.arange(len(obs_pts)), fit_idx)
x_fit, x_test = obs_pts[fit_idx], obs_pts[test_idx]
g_fit  = eval_sh(coeffs3, R3.inv().apply(x_fit))
g_test = eval_sh(coeffs3, R3.inv().apply(x_test))
print(f"||g_fit||={np.linalg.norm(g_fit):.3f}, ||g_test||={np.linalg.norm(g_test):.3f}\n")

def analyze(R_hat, label, rcond_vals=(None, 0, 1e-15, 1e-10)):
    print(f"--- {label} ---")
    A_fit, _ = sh_design(R_hat.inv().apply(x_fit), L3)
    A_te, _  = sh_design(R_hat.inv().apply(x_test), L3)
    _, _, rank, sv = np.linalg.lstsq(A_fit, g_fit, rcond=None)
    print(f"  rank={rank}/{A_fit.shape[1]}  cond={sv[0]/sv[-1]:.3e}  sv={sv[-3:]}")
    for rc in rcond_vals:
        try:
            c_hat, *_ = np.linalg.lstsq(A_fit, g_fit, rcond=rc)
            fit_res = np.linalg.norm(A_fit@c_hat - g_fit)/np.linalg.norm(g_fit)
            f_pred  = A_te @ c_hat
            cv      = np.mean((f_pred - g_test)**2) / np.mean(g_test**2)
            print(f"    rcond={str(rc):>6}: fit_res={fit_res:.3e}  cv_res={cv:.3e}  ||c||={np.linalg.norm(c_hat):.2f}")
        except Exception as e:
            print(f"    rcond={rc}: ERROR {e}")
    print()

# Test 1: correct R3 — CV MUST be ~0
analyze(R3, "CORRECT R3  (cv must be ~0)")
# Test 2: identity
analyze(Rotation.identity(), "IDENTITY")
# Test 3: random
for i, Rr in enumerate([Rotation.random(random_state=rng) for _ in range(3)]):
    analyze(Rr, f"RANDOM #{i}")

# Test 4: directly compute c via known analytic — is g_fit in col(A_fit)?
print("--- Direct membership test: is g_fit in col(A_fit)? ---")
A_fit, _ = sh_design(R3.inv().apply(x_fit), L3)   # correct R3
proj = A_fit @ np.linalg.lstsq(A_fit, g_fit, rcond=None)[0]
print(f"  ||g_fit - A_fit·c|| / ||g_fit|| = {np.linalg.norm(g_fit - proj)/np.linalg.norm(g_fit):.3e}")

A_fit_wrong, _ = sh_design(Rotation.random(random_state=rng).inv().apply(x_fit), L3)
proj_w = A_fit_wrong @ np.linalg.lstsq(A_fit_wrong, g_fit, rcond=None)[0]
print(f"  ||g_fit - A_fit_wrong·c|| / ||g_fit|| = {np.linalg.norm(g_fit - proj_w)/np.linalg.norm(g_fit):.3e}")
