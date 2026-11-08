import numpy as np
from scipy.spatial.transform import Rotation
from math import factorial

# ---------- Spherical harmonics: robust across SciPy versions ----------
# SciPy >= 1.15 removed `sph_harm`. We rebuild the *real part* of Y_l^m
# from the associated Legendre polynomial (lpmv) with a hand-coded recurrence fallback.

_HAVE_LPMV = False
try:
    from scipy.special import lpmv as _lpmv
    _HAVE_LPMV = True
except ImportError:
    pass


def _assoc_legendre(l, m, x):
    """Associated Legendre P_l^m(x) with Condon-Shortley phase; m >= 0."""
    if m > l:
        return np.zeros_like(x, dtype=float)
    if _HAVE_LPMV:
        try:
            return _lpmv(m, l, x)
        except Exception:
            pass
    # hand-coded recurrence (numerically fine for small l)
    pmm = np.ones_like(x, dtype=float)
    if m > 0:
        somx2 = np.sqrt(np.clip(1.0 - x * x, 0.0, None))
        fact = 1.0
        for _ in range(1, m + 1):
            pmm = pmm * (-fact) * somx2
            fact += 2.0
    if l == m:
        return pmm
    pmmp1 = x * (2 * m + 1) * pmm
    if l == m + 1:
        return pmmp1
    for ll in range(m + 2, l + 1):
        pll = (x * (2 * ll - 1) * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll
    return pmmp1


def _sph_real(m, l, phi, theta):
    """Real spherical harmonic Y_l^m -- full orthonormal set.
    m>0 uses cos(|m|phi), m<0 uses sin(|m|phi).  Condon-Shortley phase
    from lpmv is cancelled so that <Y_a, Y_b> = delta_ab on the sphere."""
    absm = abs(m)
    x = np.cos(theta)
    Plm = _assoc_legendre(l, absm, x) * ((-1.0) ** absm)     # kill CS phase
    norm = np.sqrt((2 * l + 1) / (4 * np.pi)
                   * factorial(l - absm) / factorial(l + absm))
    if m == 0:
        return norm * Plm
    if m > 0:
        return np.sqrt(2.0) * norm * Plm * np.cos(absm * phi)
    return np.sqrt(2.0) * norm * Plm * np.sin(absm * phi)



# ---------- sphere / strikes / field ----------
def fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(phi)], axis=1)


def random_sphere(n, rng):
    p = rng.normal(size=(n, 3))
    return p / np.linalg.norm(p, axis=1, keepdims=True)


def strike_rotation(p, omega, dt):
    L = np.cross(p, omega)
    angle = np.linalg.norm(L) * dt
    if angle < 1e-12:
        return Rotation.identity()
    axis = L / np.linalg.norm(L)
    return Rotation.from_rotvec(axis * angle)


def simulate_strikes(N, rng, scale=1.0):
    pts = random_sphere(N, rng)
    omegas = rng.normal(size=(N, 3)) * scale
    dts = rng.uniform(0.1, 1.0, N)
    R_total = Rotation.identity()
    Rs = []
    for i in range(N):
        R = strike_rotation(pts[i], omegas[i], dts[i])
        Rs.append(R)
        R_total = R * R_total
    return R_total, Rs, pts, omegas, dts


def random_sh_coeffs(L, rng):
    return {(l, m): float(rng.normal()) for l in range(L + 1) for m in range(-l, l + 1)}


def eval_sh(coeffs, pts):
    pts = np.atleast_2d(pts)
    theta = np.arccos(np.clip(pts[:, 2], -1, 1))
    phi = np.arctan2(pts[:, 1], pts[:, 0])
    vals = np.zeros(len(pts))
    for (l, m), c in coeffs.items():
        vals += c * _sph_real(m, l, phi, theta)
    return vals
