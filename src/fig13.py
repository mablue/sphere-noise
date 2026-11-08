import json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('../results/verify_final.json') as f:
    sols = json.load(f)['V3']

Dmags  = np.array([s['Dmag'] for s in sols])
absz   = np.array([abs(s['axis'][2]) for s in sols])

fig, ax = plt.subplots(figsize=(7,4.8))
# left column: |D| values; right column: |axis_z|
ax.scatter(Dmags, absz, s=90, c='steelblue', edgecolor='k', zorder=3)
# theory lines
ax.axhline(1.0, color='red',    ls='--', alpha=0.6, label='axis = ẑ  (SO(2)_z)')
ax.axhline(0.0, color='green',  ls='--', alpha=0.6, label='axis ⊥ ẑ  (π_xy)')
ax.axvline(np.pi, color='green', ls=':', alpha=0.6, label='|D| = π')
ax.set_xlabel('|D| = |R_true⁻¹ R̂|  [rad]')
ax.set_ylabel('|axis_z|')
ax.set_title('Fig 13: continuous SO(2) branch vs discrete π_xy branch')
ax.legend(fontsize=8, loc='center right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../plots/fig13_stab_structure.png', dpi=140)
print("saved plots/fig13_stab_structure.png")
