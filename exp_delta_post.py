import sys, time, json, numpy as np
from fem_core import *
from estimators import *

A = 1.1
CI = 0.493 / np.sqrt(2.0); CS = 1.0
P = 9
CHI_NY = 20

def run(Nylist, outdir, nev=14):
    exact = analytic_rect(A, 8)
    lam_ex = np.array([e[0] for e in exact])
    Vcr, Tcr, bcr = mesh_rect(A, 44, 40)
    aux, lam_cr, Hcr = cr_lower_bounds(Vcr, Tcr, nev)
    assert np.all(aux < lam_ex[:nev])
    # chi on the coarse mesh
    Nxc = int(round(1.1 * CHI_NY))
    Vc, Tc, bc = mesh_rect(A, Nxc, CHI_NY); Kc, Mc, _, _ = assemble_p1(Vc, Tc)
    lam_c, Uc, _ = eig_p1(Kc, Mc, ~bc, nev, 0.0)
    rows = []
    for Ny in Nylist:
        Nx = int(round(1.1 * Ny)); t0 = time.time()
        V, T, bnd = mesh_rect(A, Nx, Ny); free = ~bnd
        md = MeshData(V, T, bnd, deg=8); md4 = MeshData(V, T, bnd, deg=4)
        K, M, area, G = assemble_p1(V, T)
        lam, U, algres = eig_p1(K, M, free, nev, 0.0)
        h = md.h
        # prolong chi: CHI_NY -> ... -> Ny
        n_cur = CHI_NY; Uchi = Uc.copy(); idx_cur = grid_index(Vc, Nxc, CHI_NY, A)
        while n_cur < Ny:
            n_next = 2 * n_cur; nx_next = int(round(1.1 * n_next))
            Vn, Tn, bn = mesh_rect(A, nx_next, n_next); idx_n = grid_index(Vn, nx_next, n_next, A)
            Uchi = prolong_structured(idx_cur, idx_n, 2) @ Uchi; idx_cur = idx_n; n_cur = n_next
        assert n_cur == Ny and Uchi.shape[0] == len(V)
        Gex = []; Uex = []
        for (lam_e, m, n) in exact[:P + 1]:
            Ue, Ge, rq = analytic_at_quad(md, lambda X: u_rect(A, m, n, X), lambda X: grad_u_rect(A, m, n, X))
            Uex.append(Ue); Gex.append(Ge)
        per_k = {}
        sumA = sumB = sumAB = sumC = 0.0; sum_e = 0.0; sum_dist_exact = 0.0
        okB_all = True; okC_all = True; gap_all = True; sign_all = True
        for k in range(1, P + 1):
            uh = U[:, k - 1].copy(); lkh = lam[k - 1]; lk = lam_ex[k - 1]
            chi = Uchi[:, k - 1]; chi_norm = np.sqrt(chi @ (M @ chi)); ip_chi = uh @ (M @ chi)
            if ip_chi < 0: uh = -uh; ip_chi = -ip_chi           # choose the sign of u_kh so that (u_kh, chi_k) > 0
            cosk = ip_chi / chi_norm; s_k = np.sqrt(max(1.0 - cosk ** 2, 0.0))   # ||u_kh - Pi_k u_kh||, ||u_kh|| = 1
            fl = equilibrated_flux(md4, uh, lkh, k=1, check=True)
            eta_res = fl["eta_res"]
            e, eps, ip = single_errors(md, uh, Uex[k - 1], Gex[k - 1])
            lam_up_prev = lam[k - 2] if k > 1 else None
            lam_low_next = aux[k]; lam1_low = aux[0]; lamk_low = aux[k - 1]; lamk_up = lkh
            gap_ok = (lkh < lam_low_next) and (k == 1 or lam_up_prev < lkh)
            termsC = [(1 - lkh / lam_low_next) ** -1]; termsT = [lam_low_next ** -0.5 * (1 - lkh / lam_low_next) ** -1]
            if k > 1:
                termsC.append((lkh / lam_up_prev - 1) ** -1); termsT.append(lam_up_prev ** -0.5 * (lkh / lam_up_prev - 1) ** -1)
            c_k = max(termsC); ct_k = max(termsT)
            # Case A (chi variant)
            abarA = np.sqrt(2.0) * (1 - s_k) ** -0.5 * ct_k * eta_res
            etaA2 = eta_res ** 2 + (lkh + lamk_up) * abarA ** 2
            # Case B
            abarB = np.sqrt(2.0) * ct_k * eta_res
            okB = abarB <= min(np.sqrt(2 * lam1_low / lamk_up), cosk)
            etaB2 = c_k ** 2 * (1 - lamk_up * abarB ** 2 / (4 * lam1_low)) ** -1 * eta_res ** 2 if okB else np.inf
            # Case C
            abarC = np.sqrt(2.0) * c_k * CI * CS * h * eta_res
            okC = abarC <= cosk
            etaC2 = eta_res ** 2 + (lkh + lamk_up) * abarC ** 2 if okC else np.inf
            etaAB2 = min(etaA2, etaB2)
            # exact dist_a(phi_k, V_h)^2 = lam_k - ||P_h phi_k||_a^2 : from ideal_delta's machinery, recompute directly
            per_k[k] = dict(lam_k=lk, lam_kh=lkh, lam_k_low=lamk_low, lam_next_low=lam_low_next, eta_res2=eta_res ** 2, e=e, eps=eps,
                            cos_chi=cosk, s_k=s_k, c_k=c_k, ct_k=ct_k, gap_ok=bool(gap_ok),
                            abarA=abarA, etaA2=etaA2, abarB=abarB, okB=bool(okB), etaB2=etaB2, abarC=abarC, okC=bool(okC), etaC2=etaC2,
                            ratioA=etaA2 / e, ratioB=(etaB2 / e if okB else None), ratioC=(etaC2 / e if okC else None),
                            algres=float(algres[k - 1]), div_check=fl["div_check"], normal_jump=fl["normal_jump"])
            sumA += etaA2 / lamk_low; sumB += (etaB2 / lamk_low if okB else np.inf); sumAB += etaAB2 / lamk_low
            sumC += (etaC2 / lamk_low if okC else np.inf); sum_e += e / lk
            okB_all &= bool(okB); okC_all &= bool(okC); gap_all &= bool(gap_ok)
        delta_ideal = ideal_delta(md, K, free, Gex[:P], lam_ex[:P])
        Uq = np.einsum("qk,tkj->tqj", md.L, U[md.T][:, :, :P])              # discrete eigenvectors at quadrature points
        O = np.array([[np.sum(md.W * Uq[:, :, k] * Uex[j]) for j in range(P)] for k in range(P)])   # O[k,j] = (phi_kh, phi_j)
        eta2 = np.array([per_k[k]["eta_res2"] for k in range(1, P + 1)])
        R2lift = None
        byp = {}
        lamMh = lam[11]           # cluster {10,11,12}: M = 12
        kappa_plus = max(1.0, (lamMh / aux[12] - 1) ** 2)
        for p in range(1, P + 1):
            lam_next_low = aux[p]
            gap_ok = lam[p - 1] < lam_next_low
            g = (1 - lam[:p] / lam_next_low) ** -1 if gap_ok else np.full(p, np.inf)
            dhat2 = float(np.sum(g ** 2 * eta2[:p] / lam[:p]))
            C = np.array([[np.sqrt(lam_ex[j] / lam[k]) * O[k, j] for j in range(p)] for k in range(p)])   # a(w_k, psi_j), w_k = phi_kh/sqrt(lam_kh)
            smin = np.linalg.svd(C, compute_uv=False).min()
            sinT = np.sqrt(max(1 - smin ** 2, 0.0))
            did = ideal_delta(md, K, free, Gex[:p], lam_ex[:p])
            Cp = max(kappa_plus, (lamMh / aux[p] - 1) ** 2) if p <= 8 else kappa_plus
            cst = Cp / (1 - dhat2) if dhat2 < 1 else np.inf
            byp[p] = dict(lam_next_low=lam_next_low, gap_ok=bool(gap_ok), dhat=np.sqrt(dhat2), sinTheta=sinT, delta_ideal=did,
                          Cp=Cp, cst=cst, g=g.tolist(),
                          chk_ideal_le_sin=bool(did <= sinT * (1 + 1e-8) + 1e-12), chk_sin_le_dhat=bool(sinT <= np.sqrt(dhat2) * (1 + 1e-10)))
        best_p = min(byp, key=lambda p: byp[p]["cst"])
        # exact HS-type sum: sum_k dist_a(phi_k,V_h)^2/lam_k (via Galerkin projections, as in ideal_delta)
        Kf = K[free][:, free].tocsc(); lu = spla.splu(Kf); sum_dist_exact = 0.0
        for k in range(P):
            rhs = np.zeros(md.nv); contrib = np.einsum("tq,tqd,tjd->tj", md.W, Gex[k], md.G)
            np.add.at(rhs, md.T.ravel(), contrib.ravel()); x = np.zeros(md.nv); x[free] = lu.solve(rhs[free])
            sum_dist_exact += (lam_ex[k] - x @ (K @ x)) / lam_ex[k]      # ||(I-P_h)phi_k||_a^2 = lam_k - ||P_h phi_k||_a^2
        dbar = CI * CS * h * np.sqrt(lam[P - 1])
        row = dict(Ny=Ny, Nx=Nx, h=h, ndof=len(V), p=P, gap_all=gap_all, okB_all=okB_all, okC_all=okC_all,
                   delta_ideal=delta_ideal, dbar=dbar,
                   dhs_exact=np.sqrt(sum_dist_exact), dhs_e=np.sqrt(sum_e),
                   dhatA=np.sqrt(sumA), dhatB=np.sqrt(sumB), dhatAB=np.sqrt(sumAB), dhatC=np.sqrt(sumC),
                   per_k=per_k, byp=byp, best_p=best_p, cst_best=byp[best_p]["cst"], kappa_plus=kappa_plus,
                   dhat=byp[P]["dhat"], sinTheta=byp[P]["sinTheta"], cst_dhat=byp[P]["cst"],
                   chk_byp=bool(all(byp[p]["chk_ideal_le_sin"] and byp[p]["chk_sin_le_dhat"] for p in byp)),
                   chk_ideal_le_hs=bool(delta_ideal <= np.sqrt(sum_dist_exact) * (1 + 1e-10)),
                   chk_hs_le_e=bool(sum_dist_exact <= sum_e * (1 + 1e-10)),
                   chk_e_le_AB=bool(sum_e <= sumAB), chk_e_le_C=bool(sum_e <= sumC),
                   chk_e_le_eta=bool(all(per_k[k]["e"] <= per_k[k]["etaA2"] for k in per_k)))
        for key in ["dhatA", "dhatB", "dhatAB", "dhatC", "dbar"]:
            pass
        for key in ["dhatA", "dhatB", "dhatAB", "dhatC", "dbar"]:
            row["cst_" + key] = (1 - row[key] ** 2) ** -1 if row[key] < 1 else np.inf
        rows.append(row)
        print(f"Ny={Ny} done in {time.time()-t0:.1f}s: delta_ideal={delta_ideal:.4f} dhs_exact={row['dhs_exact']:.4f} dhs_e={row['dhs_e']:.4f} "
              f"dbar={dbar:.4f} dhatA={row['dhatA']:.4f} dhatB={row['dhatB']:.4f} dhatAB={row['dhatAB']:.4f} dhatC={row['dhatC']:.4f} "
              f"gap={gap_all} okB={okB_all} okC={okC_all} checks={[row[k] for k in row if k.startswith('chk_')]}")
        print("   Thm1(iv): " + "  ".join(f"p={p}: dhat={byp[p]['dhat']:.4f} sinT={byp[p]['sinTheta']:.4f} ideal={byp[p]['delta_ideal']:.4f} Cp={byp[p]['Cp']:.3f} cst={byp[p]['cst']:.4f}" for p in byp) + f"  best p={best_p}")
        sys.stdout.flush()
        with open(f"{outdir}/delta_post_rows.json", "w") as f:
            json.dump(rows, f, indent=1, default=float)
    return rows

def table(rows):
    f = lambda x: ("void" if not np.isfinite(x) else f"{x:.4f}")
    out = ["| Ny | h | δ ideal | (Σ dist²/λ)^½ | (Σ e/λ)^½ | δ̄ a priori | δ̂ Case A | δ̂ Case B | δ̂ min(A,B) | δ̂ Case C | const min(A,B) | const C | const a priori | gaps | (5.6) all | (5.9) all | checks |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        chk = all(r[k] for k in r if k.startswith("chk_"))
        out.append(f"| {r['Ny']} | {r['h']:.4f} | {r['delta_ideal']:.4f} | {r['dhs_exact']:.4f} | {r['dhs_e']:.4f} | {r['dbar']:.4f} | {f(r['dhatA'])} | {f(r['dhatB'])} | {f(r['dhatAB'])} | {f(r['dhatC'])} | {f(r['cst_dhatAB'])} | {f(r['cst_dhatC'])} | {f(r['cst_dbar'])} | {r['gap_all']} | {r['okB_all']} | {r['okC_all']} | {'ok' if chk else 'FAIL'} |")
    out.append("")
    out.append("| Ny | p | λ̲_{p+1}/π² | gap | δ ideal_p | sinΘ_a(E,W_h) | δ̂_p (Thm 1(iv)) | C_p | C_p/(1−δ̂_p²) |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        for p, q in r["byp"].items():
            out.append(f"| {r['Ny']} | {p} | {q['lam_next_low']/np.pi**2:.4f} | {q['gap_ok']} | {q['delta_ideal']:.4f} | {q['sinTheta']:.4f} | {q['dhat']:.4f} | {q['Cp']:.3f} | {f(q['cst'])} |")
    out.append("")
    out.append("| Ny | k | λ_k/π² | λ_kh/π² | gap | c_k | c̃_k | cos(u_kh,χ_k) | η_res² | e | η_A²/e | η_B²/e | η_C²/e |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        for k, q in r["per_k"].items():
            g = lambda x: ("–" if x is None else f"{x:.2f}")
            out.append(f"| {r['Ny']} | {k} | {q['lam_k']/np.pi**2:.4f} | {q['lam_kh']/np.pi**2:.4f} | {q['gap_ok']} | {q['c_k']:.2f} | {q['ct_k']:.3f} | {q['cos_chi']:.4f} | {q['eta_res2']:.3e} | {q['e']:.3e} | {g(q['ratioA'])} | {g(q['ratioB'])} | {g(q['ratioC'])} |")
    return "\n".join(out)

if __name__ == "__main__":
    Nymax = int(sys.argv[1]) if len(sys.argv) > 1 else 320
    outdir = sys.argv[2] if len(sys.argv) > 2 else "."
    rows = run([n for n in [40, 80, 160, 320] if n <= Nymax], outdir)
    print("\n" + table(rows))
    with open(f"{outdir}/delta_post_table.md", "w") as f:
        f.write(table(rows) + "\n")
