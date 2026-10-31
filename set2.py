import sys, time, json, numpy as np
from fem_core import *
from estimators import *

A = 1.1
CI = 0.493 / np.sqrt(2.0); CS = 1.0
CLUSTERS = [("high", 10, 12), ("low", 2, 3)]

def rect_mesh_fn(Nx, Ny):
    def fn(s):
        nx, ny = Nx * 2 ** s, Ny * 2 ** s
        V, T, bnd = mesh_rect(A, nx, ny)
        return V, T, bnd, grid_index(V, nx, ny, A)
    return fn

def run(Nylist, outdir, kflux=1, nev=14):
    exact = analytic_rect(A, 8)
    lam_ex = np.array([e[0] for e in exact])
    print("exact/pi^2:", np.round(lam_ex[:15] / np.pi ** 2, 4))
    # guaranteed CR lower bounds on the coarse rectangle mesh (Ny = 40)
    Vcr, Tcr, bcr = mesh_rect(A, 44, 40)
    aux, lam_cr, Hcr = cr_lower_bounds(Vcr, Tcr, nev)
    print("CR lower bounds /pi^2:", np.round(aux / np.pi ** 2, 4), " below exact:", bool(np.all(aux < lam_ex[:nev])))
    rows = []
    for Ny in Nylist:
        Nx = int(round(1.1 * Ny)); t0 = time.time()
        V, T, bnd = mesh_rect(A, Nx, Ny); free = ~bnd
        md = MeshData(V, T, bnd, deg=8); md4 = MeshData(V, T, bnd, deg=4)
        K, M, area, G = assemble_p1(V, T)
        lam, U, algres = eig_p1(K, M, free, nev, 0.0)
        h = md.h
        Uex, Gex = [], []
        for (lam_e, m, n) in exact[:nev]:
            Ue, Ge, rq = analytic_at_quad(md, lambda X: u_rect(A, m, n, X), lambda X: grad_u_rect(A, m, n, X))
            Uex.append(Ue); Gex.append(Ge)
        levels = 2 if Ny <= 160 else 1
        per_i = {}
        for i in range(1, nev):
            if not any(m_ <= i <= M_ for _, m_, M_ in CLUSTERS):
                continue
            uh = U[:, i - 1]; lih = lam[i - 1]
            fl = equilibrated_flux(md4, uh, lih, k=kflux, check=True)
            fl2 = equilibrated_flux(md4, uh, lih, k=2, check=False)
            ll = local_liftings(md, uh, lih)
            Lf, nvf = dual_norm_lower_bound(rect_mesh_fn(Nx, Ny), grid_index(V, Nx, Ny, A), levels, uh, lih)
            e, eps, ip = single_errors(md, uh, Uex[i - 1], Gex[i - 1])
            per_i[i] = dict(eta_res=fl["eta_res"], eta_res_rt2=fl2["eta_res"], R2=ll["R2"], Lf=Lf, nvf=nvf, e=e, eps=eps,
                            div_check=fl["div_check"], normal_jump=fl["normal_jump"], algres=float(algres[i - 1]))
        for cname, m_, M_ in CLUSTERS:
            J = M_ - m_ + 1; idx = list(range(m_ - 1, M_))
            E2, eps, O = cluster_quantities(md, U[:, idx], lam[idx], [Uex[j] for j in idx], lam_ex[idx])
            Qh = Q_h(lam_ex[m_ - 1], lam_ex[M_ - 1], E2, eps)
            sumR2 = sum(per_i[i]["R2"] for i in range(m_, M_ + 1))
            sumEta2 = sum(per_i[i]["eta_res"] ** 2 for i in range(m_, M_ + 1))
            sumEta2_rt2 = sum(per_i[i]["eta_res_rt2"] ** 2 for i in range(m_, M_ + 1))
            sumLf = sum(per_i[i]["Lf"] for i in range(m_, M_ + 1))
            lamMh, lammh = lam[M_ - 1], lam[m_ - 1]
            lam1_low = aux[0]; lamMp1_low = aux[M_]          # aux[M_] = lower bound of lam_{M+1}
            lam_prev_up = lam[m_ - 2] if m_ > 1 else None
            # constants
            cbar = max((lamMh / lam1_low - 1) ** 2, 1.0)
            terms = [(1 - lamMh / lamMp1_low) ** -1]
            if m_ > 1: terms.append((lammh / lam_prev_up - 1) ** -1)
            c_h = max(terms)
            gap_ok = (lamMh < lamMp1_low) and (m_ == 1 or lam_prev_up < lammh)
            loss = c_h ** 2 * (lamMh - lam1_low) ** 2 * (CI * CS * h) ** 2 / lam1_low
            p = m_ - 1
            dbar = CI * CS * h * np.sqrt(lam[p - 1]) if p > 0 else 0.0
            Cp_ok = lamMh <= 2 * lamMp1_low
            delta_ideal = ideal_delta(md, K, free, Gex[:p], lam_ex[:p])
            den = E2 + Qh
            cst_cdmsv = cbar
            cst_L2 = (1 - loss) ** -1 if loss < 1 else np.inf
            cst_new = (1 - dbar ** 2) ** -1 if dbar < 1 else np.inf
            row = dict(cluster=cname, m=m_, M=M_, J=J, Ny=Ny, Nx=Nx, h=h, ndof=len(V),
                       lam_ex=[float(x) for x in lam_ex[idx]], lam_h=[float(x) for x in lam[idx]],
                       E2=E2, eps=eps, Qh=Qh, Qh_over_E2=Qh / E2, sumR2=sumR2, sumEta2=sumEta2, sumEta2_rt2=sumEta2_rt2, sumLf=sumLf,
                       ratio_low=sumR2 / den, ratio_up=sumEta2 / den, ratio_up_rt2=sumEta2_rt2 / den, ratio_Lf=sumLf / den,
                       cbar=cbar, c_h=c_h, gap_ok=bool(gap_ok), loss=loss, dbar=dbar, delta_ideal=delta_ideal, Cp_ok=bool(Cp_ok),
                       cst_cdmsv=cst_cdmsv, cst_L2=cst_L2, cst_new=cst_new,
                       lam1_low=lam1_low, lamMp1_low=lamMp1_low, levels_lb=levels,
                       per_i={i: per_i[i] for i in range(m_, M_ + 1)},
                       chk_R2_le_new=bool(sumR2 <= cst_new * den), chk_Lf_le_new=bool(sumLf <= cst_new * den),
                       chk_R2_le_cdmsv=bool(sumR2 <= cbar * E2 + Qh), chk_Lf_le_cdmsv=bool(sumLf <= cbar * E2 + Qh),
                       chk_Lf_le_ideal=bool((1 - delta_ideal ** 2) * sumLf <= den),
                       chk_R2_le_eta2=bool(all(per_i[i]["R2"] <= per_i[i]["eta_res"] ** 2 * (1 + 1e-12) for i in range(m_, M_ + 1))),
                       chk_eps_pos=bool(eps >= -1e-12), chk_E2_pos=bool(E2 >= -1e-12))
            rows.append(row)
        print(f"Ny={Ny} done in {time.time() - t0:.1f}s; lam_h/pi^2={np.round(lam[:13] / np.pi ** 2, 3)}")
        sys.stdout.flush()
    with open(f"{outdir}/set2_rows.json", "w") as f:
        json.dump(rows, f, indent=1, default=float)
    return rows

def table(rows):
    out = ["| cluster | Ny | h | E² | ε | Q_h/E² | ratio: ΣR²/(E²+Q_h) | ΣL_f/(E²+Q_h) | Ση²/(E²+Q_h) | c̄_h (CDMSV) | loss (L²) | const L² | δ̄ (new) | δ ideal | const new | gap | C_p=1 | checks |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        chk = all(r[k] for k in r if k.startswith("chk_"))
        cl = ("– (void)" if not np.isfinite(r["cst_L2"]) else f"{r['cst_L2']:.4f}")
        out.append(f"| {r['cluster']} {{{r['m']}..{r['M']}}} | {r['Ny']} | {r['h']:.4f} | {r['E2']:.3e} | {r['eps']:.2e} | {r['Qh_over_E2']:.1e} | {r['ratio_low']:.4f} | {r['ratio_Lf']:.4f} | {r['ratio_up']:.4f} | {r['cbar']:.2f} | {r['loss']:.3f} | {cl} | {r['dbar']:.4f} | {r['delta_ideal']:.4f} | {r['cst_new']:.5f} | {r['gap_ok']} | {r['Cp_ok']} | {'ok' if chk else 'FAIL'} |")
    return "\n".join(out)

if __name__ == "__main__":
    Nymax = int(sys.argv[1]) if len(sys.argv) > 1 else 320
    outdir = sys.argv[2] if len(sys.argv) > 2 else "."
    Nylist = [n for n in [40, 80, 160, 320] if n <= Nymax]
    rows = run(Nylist, outdir)
    print("\n" + table(rows))
    with open(f"{outdir}/set2_table.md", "w") as f:
        f.write(table(rows) + "\n")
