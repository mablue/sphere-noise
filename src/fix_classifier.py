# Just so the record shows: all 40 solutions are in Stab
import json, numpy as np
with open('../results/phase5_final.json') as f:
    d = json.load(f)
z = np.array([0.,0.,1.])
n_in_stab = 0
for s in d['B']['sols']:
    ax = np.array(s['Delta_axis']); mag = s['Delta_mag']
    cos_par = abs(np.dot(ax, z))
    ok = (cos_par > 0.9) or (cos_par < 0.1 and abs(mag-np.pi) < 0.1)
    n_in_stab += int(ok)
print(f"Corrected classifier: {n_in_stab}/{len(d['B']['sols'])} solutions in Stab(Y_2^0)")
