import sys, time, json, numpy as np
from fem_core import *
from estimators import *
from liftings_pk import local_liftings_pk, with_gauss_quadrature
import set2 as S2
from set1 import G_fun, eta_511, AUX_PAPER, CI, CS

outdir = sys.argv[1] if len(sys.argv) > 1 else "."
log = open(f"{outdir}/log.txt", "w")
def P(*a):
    s = " ".join(str(x) for x in a); print(s); log.write(s + "\n"); log.flush()

P("=== A. Set 2, Ny = 10, 20 ===")
rowsA = S2.run([10, 20], outdir)
import os; os.rename(f"{outdir}/set2_rows.json", f"{outdir}/set2_coarse_rows.json")
P(S2.table(rowsA))
P("\nconditions per row (unrounded):")
for r in rowsA:
    P(f"  {r['cluster']} Ny={r['Ny']}: lam_h/pi^2={np.round(np.array(r['lam_h'])/np.pi**2,4)} lamMh/lamMp1_low={r['lam_h'][-1]/r['lamMp1_low']:.4f} "
      f"gap_ok={r['gap_ok']} Cp_ok={r['Cp_ok']} dbar={r['dbar']:.5f} delta_ideal={r['delta_ideal']:.5f} loss={r['loss']:.4f} "
      f"loss/dbar^2={r['loss']/r['dbar']**2 if r['dbar']>0 else float('nan'):.1f} cst_new={r['cst_new']:.5f} cst_L2={r['cst_L2']} cbar={r['cbar']:.3f} "
      f"ratio_low={r['ratio_low']:.4f} ratio_up={r['ratio_up']:.4f} ratio_Lf={r['ratio_Lf']:.4f} E2={r['E2']:.4e} eps={r['eps']:.3e} Qh/E2={r['Qh_over_E2']:.3e} checks={all(r[k] for k in r if k.startswith('chk_'))}")

P("\n=== B. Assumption 2.6: overlap matrix O_ik = (phi_ih, phi_k), i,k in cluster ===")
A = S2.A
exact = analytic_rect(A, 8); lam_ex = np.array([e[0] for e in exact])
resB = []
for Ny in [10, 20, 40, 80, 160, 320]:
    Nx = int(round(1.1 * Ny)); t0 = time.time()
    V, T, bnd = mesh_rect(A, Nx, Ny); free = ~bnd
    md = MeshData(V, T, bnd, deg=8)
    K, M, area, G = assemble_p1(V, T)
    lam, U, algres = eig_p1(K, M, free, 14, 0.0)
    Uex = []
    for (lam_e, m, n) in exact[:14]:
        Ue, Ge, rq = analytic_at_quad(md, lambda X: u_rect(A, m, n, X), lambda X: grad_u_rect(A, m, n, X))
        Uex.append(Ue)
    for cname, m_, M_ in S2.CLUSTERS:
        idx = list(range(m_ - 1, M_))
        E2, eps, O = cluster_quantities(md, U[:, idx], lam[idx], [Uex[j] for j in idx], lam_ex[idx])
        sv = np.linalg.svd(O, compute_uv=False)
        # also the overlap of the discrete cluster with the 13 lowest exact eigenfunctions (leakage outside the cluster)
        Ofull = np.array([[np.sum(md.W * np.einsum("qk,tk->tq", md.L, U[:, i][md.T]) * Uex[j]) for j in range(14)] for i in idx])
        leak = np.sqrt(max(0.0, np.sum(Ofull ** 2) - np.sum(O ** 2)))
        resB.append(dict(cluster=cname, Ny=Ny, J=len(idx), sv=sv.tolist(), sigma_min=float(sv.min()), sigma_max=float(sv.max()),
                         O=O.tolist(), leak_outside_cluster=float(leak), eps=float(eps), algres=float(algres[idx].max())))
        P(f"  {cname} Ny={Ny}: singular values of O = {np.array2string(sv, precision=6)}  sigma_min={sv.min():.6f}  "
          f"sqrt(sum O^2 outside cluster, first 14 exact)={leak:.3e}  eps={eps:.3e}  t={time.time()-t0:.1f}s")
with open(f"{outdir}/assumption26_rows.json", "w") as f:
    json.dump(resB, f, indent=1)

P("\n=== C. Set 1, N = 40, i = 1..4, aux = paper: reconstruction sensitivity ===")
exactT = analytic_triangle(6); lamT = np.array([e[0] for e in exactT])
N = 40
V, T, bnd = mesh_triangle(N); free = ~bnd
md = MeshData(V, T, bnd, deg=8); md4 = MeshData(V, T, bnd, deg=4); mdg = with_gauss_quadrature(md, 10)
K, M, area, G = assemble_p1(V, T)
lam, U, algres = eig_p1(K, M, free, 6, 40.0)
h = md.h
TABLE7 = {1: (1.22, 1.11), 2: (2.39, 3.36), 3: (2.77, None), 4: (6.97, None)}   # (I^ub, I^lb) of 2017 Table 7, N = 40 rows (None = void)
rowsC = []
for i in range(1, 5):
    uh = U[:, i - 1]; lih = lam[i - 1]; li = lamT[i - 1]; g = lih - li
    fluxes = {"RT1(deg4)": equilibrated_flux(md4, uh, lih, k=1, check=True)["eta_res"],
              "RT1(deg8)": equilibrated_flux(md, uh, lih, k=1, check=True)["eta_res"],
              "RT2(deg4)": equilibrated_flux(md4, uh, lih, k=2, check=True)["eta_res"]}
    lifts = {"P2(deg8)": local_liftings(md, uh, lih)["R2"],
             "P2(gauss10)": local_liftings_pk(mdg, uh, lih, k=2)["R2"],
             "P3(deg8)": local_liftings_pk(md, uh, lih, k=3)["R2"],
             "P3(gauss10)": local_liftings_pk(mdg, uh, lih, k=3)["R2"]}
    aux = AUX_PAPER; lam1_low = aux[0]; lami_low = aux[i - 1]; lamip1_low = aux[i]
    lam_up_prev = lam[i - 2] if i > 1 else None
    terms = [(1 - lih / lamip1_low) ** -1]
    if i > 1: terms.append((lih / lam_up_prev - 1) ** -1)
    c_ih = max(terms)
    cbar = max((lih / lam1_low - 1) ** 2, 1.0)
    P(f"\n i={i}: lam_ih={lih:.6f} g={g:.6e} c_ih={c_ih:.4f} cbar={cbar:.4f}  eta_res: " + ", ".join(f"{k}={v:.6e}" for k, v in fluxes.items())
      + "  R2: " + ", ".join(f"{k}={v:.6e}" for k, v in lifts.items()))
    for fk, eta_res in fluxes.items():
        alpha = np.sqrt(2.0) * c_ih * CI * CS * h * eta_res
        eta_i2 = eta_res ** 2 + (lih + lih) * alpha ** 2          # 2017 (5.5): (lam_ih + lam_i_up) abar^2 with lam_i_up = lam_ih
        Iub = eta_i2 / g
        Tt = lih * alpha ** 2
        for lk, R2 in lifts.items():
            et_orig = eta_511(lami_low, cbar, R2, lih, alpha ** 2)
            Ilb = g / et_orig if et_orig > 0 else np.inf
            dbar = CI * CS * h * np.sqrt(lam[i - 2]) if i > 1 else 0.0
            et_new = max(-Tt + G_fun(1.0, (1 - dbar ** 2) * R2, lami_low), 0.0) if dbar < 1 else 0.0
            Mbar = (lih - lam1_low) ** 2 / lam1_low
            et_L2 = max(-Tt + G_fun(1.0, R2 - Mbar * alpha ** 2, lami_low), 0.0)
            t7 = TABLE7[i]
            dev_ub = (Iub - t7[0]) / t7[0] * 100
            dev_lb = ((Ilb - t7[1]) / t7[1] * 100) if (t7[1] is not None and np.isfinite(Ilb)) else None
            rowsC.append(dict(i=i, flux=fk, lift=lk, eta_res=eta_res, R2=R2, alpha=alpha, T=Tt, eta_i2=eta_i2, Iub=Iub, et_orig=et_orig,
                              Ilb_orig=Ilb, et_new=et_new, Ilb_new=(g / et_new if et_new > 0 else np.inf), et_L2=et_L2,
                              Ilb_L2=(g / et_L2 if et_L2 > 0 else np.inf), dev_Iub_pct=dev_ub, dev_Ilb_pct=dev_lb, g=g, lam_ih=lih))
            P(f"   flux={fk:10s} lift={lk:12s}  T={Tt:.6e}  eta~2(5.11)={et_orig:.6e}  I^lb={Ilb:.4f} (Table7 {t7[1]}, dev {dev_lb if dev_lb is None else round(dev_lb,2)}%)"
              f"  I^ub={Iub:.4f} (Table7 {t7[0]}, dev {dev_ub:+.2f}%)  eta~2(new)={et_new:.4e}  eta~2(L2)={et_L2:.4e}")
with open(f"{outdir}/set1_N40_sensitivity.json", "w") as f:
    json.dump(rowsC, f, indent=1, default=float)
P("\ndone.")
