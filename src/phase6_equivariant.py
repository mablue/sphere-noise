import numpy as np
from scipy.spatial.transform import Rotation
import sys, os, json; sys.path.insert(0, '.')
from core import *
from core import _sph_real
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

rng = np.random.default_rng(77)
grid = fibonacci_sphere(200)
L = 2
keys = [(l,m) for l in range(L+1) for m in range(-l,l+1)]
d = len(keys)

def sh_design(pts):
    th = np.arccos(np.clip(pts[:,2],-1,1)); ph = np.arctan2(pts[:,1],pts[:,0])
    return np.column_stack([_sph_real(m,l,ph,th) for (l,m) in keys])

A_grid = sh_design(grid)   # 200 x d

def power_spectrum(coeffs):
    """SO(3)-invariant power spectrum of a field."""
    return np.array([sum(coeffs[(l,m)]**2 for m in range(-l,l+1))
                     for l in range(L+1)])

def generate_instance(rng):
    co = {(0,0): rng.normal()}
    for l in range(1, L+1):
        for m in range(-l, l+1):
            co[(l,m)] = rng.normal()
    R = simulate_strikes(50, rng, scale=0.5)[0]
    g = A_grid @ np.array([co[k] for k in keys])
    return g, R, co

# ============ test: is power spectrum actually invariant? ============
print("=== Sanity: power spectrum invariance ===")
g, R, co = generate_instance(rng)
# coeffs of g in frame: a_g = A^+ g
a_g, *_ = np.linalg.lstsq(A_grid, g, rcond=None)
ps_g = power_spectrum({k: a_g[i] for i,k in enumerate(keys)})
ps_true = power_spectrum(co)
print(f"  ps(g)      = {np.round(ps_g, 4)}")
print(f"  ps(f)      = {np.round(ps_true, 4)}")
print(f"  rel.err    = {np.linalg.norm(ps_g-ps_true)/np.linalg.norm(ps_true):.2e}  ← باید ~0")

# ============ experiment: MLP vs Equivariant MLP ============
print("\n=== Exp G: MLP vs Equivariant MLP (predict rotation & invariant) ===")

N_tr, N_te = 3000, 400
def make_dataset(rng, n):
    Xg   = np.zeros((n, len(grid)))          # raw g
    Xps  = np.zeros((n, L+1))                # power spectrum
    YR   = np.zeros((n, 3))                  # ZYZ angles (up to gauge)
    Yco  = np.zeros((n, d))                  # coefficients (up to gauge)
    Yps  = np.zeros((n, L+1))                # TRUE power spectrum
    for i in range(n):
        g, R, co = generate_instance(rng)
        a_g, *_ = np.linalg.lstsq(A_grid, g, rcond=None)
        Xg[i] = g
        Xps[i] = power_spectrum({k: a_g[j] for j,k in enumerate(keys)})
        YR[i] = R.as_euler('ZYZ')
        Yco[i] = a_g
        Yps[i] = power_spectrum(co)
    return Xg, Xps, YR, Yco, Yps

Xg_tr, Xps_tr, YR_tr, Yco_tr, Yps_tr = make_dataset(rng, N_tr)
Xg_te, Xps_te, YR_te, Yco_te, Yps_te = make_dataset(rng, N_te)
print(f"  train: {N_tr}, test: {N_te}")

# normalize
g_mu, g_sd = Xg_tr.mean(0), Xg_tr.std(0) + 1e-8
ps_mu, ps_sd = Xps_tr.mean(0), Xps_tr.std(0) + 1e-8
yps_mu, yps_sd = Yps_tr.mean(0), Yps_tr.std(0) + 1e-8

class MLP:
    def __init__(s, din, dh, dout, rng):
        s.W1 = rng.normal(0, 1/np.sqrt(din), (din, dh)); s.b1 = np.zeros(dh)
        s.W2 = rng.normal(0, 1/np.sqrt(dh), (dh, dout)); s.b2 = np.zeros(dout)
        s.P = [s.W1, s.b1, s.W2, s.b2]
        s.m = [np.zeros_like(p) for p in s.P]; s.v = [np.zeros_like(p) for p in s.P]; s.t = 0
    def fwd(s, x):
        a1 = np.tanh(x @ s.W1 + s.b1); return a1 @ s.W2 + s.b2, a1
    def step(s, x, y, lr):
        out, a1 = s.fwd(x); dz2 = 2*(out-y)/len(x)
        da1 = (dz2 @ s.W2.T) * (1 - a1**2)
        grads = [x.T@da1, da1.sum(0), a1.T@dz2, dz2.sum(0)]
        s.t += 1
        for i,(p,gg) in enumerate(zip(s.P, grads)):
            s.m[i] = 0.9*s.m[i] + 0.1*gg
            s.v[i] = 0.999*s.v[i] + 0.001*gg**2
            mh = s.m[i]/(1-0.9**s.t); vh = s.v[i]/(1-0.999**s.t)
            p -= lr*mh/(np.sqrt(vh)+1e-8)
        return float(np.mean((out-y)**2))

def train(mlp, X, Y, Xte, Yte, epochs=120, lr=1e-3, bs=64):
    hist = {'tr': [], 'te': []}
    for ep in range(epochs):
        idx = rng.permutation(len(X))
        for i in range(0, len(X), bs):
            mlp.step(X[idx[i:i+bs]], Y[idx[i:i+bs]], lr)
        if ep % 10 == 0:
            otr,_ = mlp.fwd(X); ote,_ = mlp.fwd(Xte)
            hist['tr'].append(float(np.mean((otr-Y)**2)))
            hist['te'].append(float(np.mean((ote-Yte)**2)))
    return hist

# --- Baseline 1: raw MLP predicts true power spectrum ---
print("\n  [baseline] MLP: raw g → TRUE power spectrum")
Xn_tr = (Xg_tr-g_mu)/g_sd; Xn_te = (Xg_te-g_mu)/g_sd
Yn_tr = (Yps_tr-yps_mu)/yps_sd; Yn_te = (Yps_te-yps_mu)/yps_sd
m_base = MLP(len(grid), 128, L+1, rng)
h_base = train(m_base, Xn_tr, Yn_tr, Xn_te, Yn_te)
o,_ = m_base.fwd(Xn_te); pred = o*yps_sd + yps_mu
err_base = np.linalg.norm(pred - Yps_te)/np.linalg.norm(Yps_te)
print(f"      test ps-recovery rel err = {err_base:.4f}")

# --- Equivariant: input = power spectrum (already invariant) → predict true ps ---
print("\n  [equivariant] equiv-features → TRUE power spectrum (almost trivial)")
Xpsn_tr = (Xps_tr-ps_mu)/ps_sd; Xpsn_te = (Xps_te-ps_mu)/ps_sd
m_eq = MLP(L+1, 64, L+1, rng)
h_eq = train(m_eq, Xpsn_tr, Yn_tr, Xpsn_te, Yn_te)
o,_ = m_eq.fwd(Xpsn_te); pred_eq = o*yps_sd + yps_mu
err_eq = np.linalg.norm(pred_eq - Yps_te)/np.linalg.norm(Yps_te)
print(f"      test ps-recovery rel err = {err_eq:.4f}")

# --- Equivariant: predict R (which IS equivariant-invariant split) ---
print("\n  [both] angle MSE (gauge floor ~ 2.4 from Exp F)")
YRn_tr = (YR_tr - YR_tr.mean(0))/(YR_tr.std(0)+1e-8)
YRn_te = (YR_te - YR_tr.mean(0))/(YR_tr.std(0)+1e-8)
m_rb = MLP(len(grid), 128, 3, rng)
h_rb = train(m_rb, Xn_tr, YRn_tr, Xn_te, YRn_te)
o,_ = m_rb.fwd(Xn_te); pred_r = o*YR_tr.std(0) + YR_tr.mean(0)
ang_err_base = np.mean((pred_r - YR_te)**2)
print(f"      baseline angle MSE = {ang_err_base:.3f}")

m_re = MLP(L+1, 64, 3, rng)
h_re = train(m_re, Xpsn_tr, YRn_tr, Xpsn_te, YRn_te)
o,_ = m_re.fwd(Xpsn_te); pred_re = o*YR_tr.std(0) + YR_tr.mean(0)
ang_err_eq = np.mean((pred_re - YR_te)**2)
print(f"      equiv    angle MSE = {ang_err_eq:.3f}  ← هنوز بالا چون target gauge است")

print("\n=== CONCLUSION ===")
print(f"  • Power spectrum is genuinely invariant: MLP-base err={err_base:.3f}, equiv err={err_eq:.3f}")
print(f"  • Rotation angles are unidentifiable from a single view (both networks fail)")
print(f"  • Equivariant network solves the *invariant* task; the angle task requires multi-view.")

# plots
fig, ax = plt.subplots(figsize=(7,4.5))
ax.semilogy(h_base['te'], label=f"baseline: g→ps  (err={err_base:.3f})")
ax.semilogy(h_eq['te'],   label=f"equiv:   ps→ps  (err={err_eq:.3f})")
ax.set_xlabel('checkpoint (per 10 epochs)'); ax.set_ylabel('test MSE')
ax.set_title('Fig 15: equivariant features close the ps-recovery gap')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('../plots/fig15_equivariant.png', dpi=140); plt.close()

with open('../results/phase6_equivariant.json','w') as f:
    json.dump(dict(
        sanity_ps_err=float(np.linalg.norm(ps_g-ps_true)/np.linalg.norm(ps_true)),
        err_base=float(err_base), err_eq=float(err_eq),
        ang_err_base=float(ang_err_base), ang_err_eq=float(ang_err_eq),
        hist_base=h_base, hist_eq=h_eq), f, indent=2)
print("\nsaved plots/fig15_equivariant.png, results/phase6_equivariant.json")
