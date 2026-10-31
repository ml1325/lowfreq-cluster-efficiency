import sys, time, json, numpy as np
import scipy.sparse as sp, scipy.sparse.linalg as spla
from fem_core import *
from estimators import *

A = 1.1
P = 9
KFLUX = 1

def rect_mesh_fn(Nx, Ny):
    def fn(s):
        nx, ny = Nx * 2 ** s, Ny * 2 ** s
        V, T, bnd = mesh_rect(A, nx, ny)
        return V, T, bnd, grid_index(V, nx, ny, A)
    return fn

def flux_at_quad(md, sigma, k):
    nt, nq = md.nt, md.nq
    nb = (k + 1) * (k + 3)
    Xi = (md.X - md.cK[:, None, :]) / md.sK[:, None, None]
    Phi, _ = rt_eval(k, Xi.reshape(-1, 2))
    Phi = Phi.reshape(nt, nq, nb, 2)
    return np.einsum("tj,tqjd->tqd", sigma, Phi)

def run(Nylist, outdir, nev=14):
    exact = analytic_rect(A, 8)
    lam_ex = np.array([e[0] for e in exact])
    Vcr, Tcr, bcr = mesh_rect(A, 44, 40)
    aux, lam_cr, Hcr = cr_lower_bounds(Vcr, Tcr, nev)
    assert np.all(aux < lam_ex[:nev])
    rows = []
    for Ny in Nylist:
        Nx = int(round(1.1 * Ny)); t0 = time.time()
        V, T, bnd = mesh_rect(A, Nx, Ny); free = ~bnd
        md4 = MeshData(V, T, bnd, deg=4)
        K, M, area, G = assemble_p1(V, T)
        lam, U, algres = eig_p1(K, M, free, nev, 0.0)
        lam_low_next = aux[P]                      # guaranteed lower bound of lambda_{p+1} (CR, Ny = 40 mesh)
        assert lam[P - 1] < lam_low_next
        s_gap = 1.0 / lam[P - 1] - 1.0 / lam_low_next
        g = (1 - lam[:P] / lam_low_next) ** -1
        # fluxes and q_k at quadrature points
        Q = np.zeros((P, md4.nt, md4.nq, 2)); eta2 = np.zeros(P); divchk = np.zeros(P); jmp = np.zeros(P)
        for k in range(P):
            uh = U[:, k]; lkh = lam[k]
            fl = equilibrated_flux(md4, uh, lkh, k=KFLUX, check=True)
            Sq = flux_at_quad(md4, fl["sigma"], KFLUX)
            Gu = np.einsum("tki,tk->ti", md4.G, uh[md4.T])
            Q[k] = (Gu[:, None, :] + Sq) / lkh ** 1.5
            eta2[k] = fl["eta_res"] ** 2; divchk[k] = fl["div_check"]; jmp[k] = fl["normal_jump"]
        Gram_q = np.einsum("tq,ktqd,ltqd->kl", md4.W, Q, Q)
        chk_diag = np.abs(np.diag(Gram_q) * lam[:P] ** 3 - eta2).max() / eta2.max()
        ev_q = np.linalg.eigvalsh(Gram_q)
        U_lin = np.sqrt(ev_q.max()) / s_gap
        HS = np.sqrt(np.trace(Gram_q)) / s_gap
        per_mode_HS2 = np.diag(Gram_q) / s_gap ** 2
        # fine-mesh Galerkin liftings of the p residuals (lower bounds of ||r_k||_a and of the operator norm)
        levels = 2 if Ny <= 160 else 1
        fn = rect_mesh_fn(Nx, Ny)
        Uf = U[:, :P].copy(); idx_prev = grid_index(V, Nx, Ny, A)
        for s in range(1, levels + 1):
            Vf, Tf, bf, idx = fn(s)
            Pm = prolong_structured(idx_prev, idx, 2)
            Uf = Pm @ Uf; idx_prev = idx
        Kf, Mf, _, _ = assemble_p1(Vf, Tf); freef = ~bf
        Res = (Mf @ Uf) * lam[None, :P] - Kf @ Uf               # columns: residual vectors on the fine mesh
        lu = spla.splu(Kf[freef][:, freef].tocsc())
        Wf = lu.solve(Res[freef])                                # Galerkin liftings (free dofs)
        Gram_R = Res[freef].T @ Wf                               # a(P_f R_k, P_f R_l) in dual-norm scaling
        Gram_R = 0.5 * (Gram_R + Gram_R.T)
        scale = lam[:P] ** -1.5
        Gram_r = scale[:, None] * Gram_R * scale[None, :]       # a(P_f r_k, P_f r_l)
        ev_r = np.linalg.eigvalsh(Gram_r)
        L_gram = np.sqrt(max(ev_r.max(), 0.0)) / s_gap
        L_1 = np.sqrt(Gram_r[0, 0]) / s_gap
        L_modes = np.sqrt(np.diag(Gram_r)) / s_gap
        Lf = np.diag(Gram_R)                                     # lower bounds of ||R_k||_{-1}^2
        # ours
        dhat = np.sqrt(np.sum(g ** 2 * eta2 / lam[:P]))
        dhat_L = np.sqrt(np.sum(g ** 2 * Lf / lam[:P]))
        chk_L_le_eta = bool(np.all(Lf <= eta2 * (1 + 1e-10)))
        chk_gram = bool(ev_r.max() <= ev_q.max() * (1 + 1e-10))
        row = dict(Ny=Ny, Nx=Nx, h=md4.h, ndof=len(V), ndof_fine=len(Vf), levels=levels, p=P,
                   lam_ph=lam[P - 1], lam_low_next=lam_low_next, s_gap=s_gap,
                   lam_h=lam[:P].tolist(), eta2=eta2.tolist(), Lf=Lf.tolist(), ratio_Lf_eta2=(Lf / eta2).tolist(), g=g.tolist(),
                   div_check=divchk.tolist(), normal_jump=jmp.tolist(), chk_diag=chk_diag,
                   dhat=dhat, dhat_L=dhat_L,
                   LV_upper_lin=U_lin, LV_HS=HS, LV_per_mode_HS2=per_mode_HS2.tolist(),
                   LV_lower_gram=L_gram, LV_lower_mode1=L_1, LV_lower_modes=L_modes.tolist(),
                   ev_q=ev_q.tolist(), ev_r=ev_r.tolist(),
                   chk_L_le_eta=chk_L_le_eta, chk_gram=chk_gram,
                   chk_order=bool(L_1 <= L_gram * (1 + 1e-10) and L_gram <= U_lin * (1 + 1e-10) and U_lin <= HS * (1 + 1e-10)))
        rows.append(row)
        print(f"Ny={Ny} ({time.time()-t0:.0f}s, fine ndof {len(Vf)}): s={s_gap:.4e}  ours dhat={dhat:.4f} (dhat_L={dhat_L:.4f})  "
              f"LV(45): lower mode1={L_1:.4f} lower gram={L_gram:.4f}  upper lin={U_lin:.4f}  HS={HS:.4f}  "
              f"chk diag={chk_diag:.1e} L<=eta={chk_L_le_eta} gram={chk_gram} order={row['chk_order']}")
        print("   per-mode LV HS^2 terms:", np.round(per_mode_HS2, 4).tolist())
        print("   per-mode lower bounds :", np.round(L_modes, 4).tolist(), " Lf/eta2:", np.round(Lf / eta2, 4).tolist())
        sys.stdout.flush()
        with open(f"{outdir}/lv_compare_rows.json", "w") as f:
            json.dump(rows, f, indent=1, default=float)
    return rows

def table(rows):
    out = ["| Ny | h | σ = 1/λ_9h − 1/λ̲_10 | ours δ̂ | ours with fine-mesh ‖R_k‖ (lower) | LV (45) lower (mode 1) | LV (45) lower (Gram) | LV (45) upper (linear flux) | LV HS form | min ratio (45)/δ̂ | checks |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        chk = r["chk_L_le_eta"] and r["chk_gram"] and r["chk_order"] and r["chk_diag"] < 1e-8
        out.append(f"| {r['Ny']} | {r['h']:.4f} | {r['s_gap']:.4e} | {r['dhat']:.4f} | {r['dhat_L']:.4f} | {r['LV_lower_mode1']:.4f} | {r['LV_lower_gram']:.4f} | {r['LV_upper_lin']:.4f} | {r['LV_HS']:.4f} | {r['LV_lower_gram']/r['dhat']:.2f} | {'ok' if chk else 'FAIL'} |")
    out.append("")
    out.append("| Ny | k | λ_kh/π² | g_k | η_k² | L_f,k (fine-mesh lower bound of ‖R_k‖²₋₁) | L_f,k/η_k² | LV per-mode HS² term | per-mode lower bound ‖P_f r_k‖/σ |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        for k in range(r["p"]):
            out.append(f"| {r['Ny']} | {k+1} | {r['lam_h'][k]/np.pi**2:.4f} | {r['g'][k]:.3f} | {r['eta2'][k]:.3e} | {r['Lf'][k]:.3e} | {r['ratio_Lf_eta2'][k]:.4f} | {r['LV_per_mode_HS2'][k]:.4f} | {r['LV_lower_modes'][k]:.4f} |")
    return "\n".join(out)

if __name__ == "__main__":
    Nymax = int(sys.argv[1]) if len(sys.argv) > 1 else 320
    outdir = sys.argv[2] if len(sys.argv) > 2 else "."
    rows = run([n for n in [40, 80, 160, 320] if n <= Nymax], outdir)
    print("\n" + table(rows))
    with open(f"{outdir}/lv_compare_table.md", "w") as f:
        f.write(table(rows) + "\n")
