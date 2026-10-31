import time, numpy as np
from fem_core import *
from estimators import *
N = 20
V, T, bnd = mesh_triangle(N)
md = MeshData(V, T, bnd)
K, M, area, G = assemble_p1(V, T)
free = ~bnd
lam, U, res = eig_p1(K, M, free, 6, 40.0)
print("lam_h", lam, "alg res", res.max())
pairs = analytic_triangle(6)
print(pairs[:5])
for i in [1, 3, 4]:
    uh = U[:, i-1]; lh = lam[i-1]
    t0 = time.time(); fl = equilibrated_flux(md, uh, lh, k=2); t1 = time.time()
    ll = local_liftings(md, uh, lh); t2 = time.time()
    print(f"i={i}: eta_res={fl['eta_res']:.5e} R={np.sqrt(ll['R2']):.5e}  div_chk={fl['div_check']:.2e} jump={fl['normal_jump']:.2e} cons={fl['max_constraint_resid']:.2e} shapes={fl['n_patch_shapes']} t_flux={t1-t0:.2f}s t_lift={t2-t1:.2f}s  patchE={ll['sum_patch_energy']:.4e} gr2={ll['grad_r2']:.4e}")
