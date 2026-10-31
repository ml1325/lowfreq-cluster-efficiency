import sys, time, json, numpy as np
from fem_core import *
from estimators import *

CI = 0.493 / np.sqrt(2.0); CS = 1.0
AUX_PAPER = np.array([49.2883, 98.4296, 127.937, 166.975, 196.439])      # 2017 Section 7.4 (Liu's method, 2145 triangles)

def G_fun(kappa, S, lam_low):
    if S <= 0: return 0.0
    return 2.0 * S / (kappa + np.sqrt(kappa ** 2 + 4.0 * S / lam_low))

def eta_511(lam_low, cbar, R2, lam_up, alpha2):
    d = lam_low ** 2 * cbar ** 2 + 4.0 * lam_low * R2
    return max(-lam_up * alpha2 + 0.5 * (np.sqrt(d) - lam_low * cbar), 0.0)

def tri_mesh_fn(N):
    def fn(s):
        n = N * 2 ** s
        V, T, bnd = mesh_triangle(n)
        return V, T, bnd, grid_index(V, n, n)
    return fn

def run(Nlist, outdir, imax=4, levels_lb=2, chi_N=20):
    exact = analytic_triangle(6)                      # (lam, m, n) sorted
    lam_ex = np.array([e[0] for e in exact])
    # sign characterisation chi_i: coarse-mesh discrete eigenvectors (N = chi_N), prolonged to the fine meshes
    Vc, Tc, bc = mesh_triangle(chi_N); Kc, Mc, _, _ = assemble_p1(Vc, Tc)
    lam_c, Uc, _ = eig_p1(Kc, Mc, ~bc, 6, 40.0)
    # CR lower bounds on a coarse mesh with ~2145 triangles (N = 46 -> 2116)
    Vcr, Tcr, bcr = mesh_triangle(46)
    aux_cr, lam_cr, Hcr = cr_lower_bounds(Vcr, Tcr, 6)
    print("CR lower bounds (N=46):", np.round(aux_cr[:5], 4), " paper:", AUX_PAPER)
    rows = []
    for N in Nlist:
        t0 = time.time()
        V, T, bnd = mesh_triangle(N); free = ~bnd
        md = MeshData(V, T, bnd, deg=8); md4 = MeshData(V, T, bnd, deg=4)
        K, M, area, G = assemble_p1(V, T)
        lam, U, algres = eig_p1(K, M, free, 6, 40.0)
        h = md.h
        # chi on this mesh: prolong the coarse eigenvectors (nested: N multiple of chi_N by powers of 2)
        idx_c = grid_index(Vc, chi_N, chi_N); n_cur = chi_N; Uchi = Uc.copy(); idx_cur = idx_c
        while n_cur < N:
            n_next = 2 * n_cur
            Vn, Tn, bn = mesh_triangle(n_next); idx_n = grid_index(Vn, n_next, n_next)
            Uchi = prolong_structured(idx_cur, idx_n, 2) @ Uchi; idx_cur = idx_n; n_cur = n_next
        assert n_cur == N
        Uex = []; Gex = []
        for (lam_e, m, n) in exact[:imax + 1]:
            Ue, Ge, rq = analytic_at_quad(md, lambda X: u_triangle(m, n, X), lambda X: grad_u_triangle(m, n, X))
            Uex.append(Ue); Gex.append(Ge)
        for i in range(1, imax + 1):
            uh = U[:, i - 1]; lih = lam[i - 1]; li = lam_ex[i - 1]; g = lih - li
            fl = equilibrated_flux(md4, uh, lih, k=1, check=True)      # RT1 as in 2017 Section 7 (div sigma = lam u_h exactly for P1)
            fl2 = equilibrated_flux(md4, uh, lih, k=2, check=True)     # RT2 (= p+1) variant, reported only
            ll = local_liftings(md, uh, lih)
            eta_res = fl["eta_res"]; R2 = ll["R2"]; eta_res_rt2 = fl2["eta_res"]
            e, eps, ip = single_errors(md, uh, Uex[i - 1], Gex[i - 1])
            # fine-mesh lower bound of ||Res||_{-1}^2
            try:
                Lf, nvf = dual_norm_lower_bound(tri_mesh_fn(N), grid_index(V, N, N), levels_lb, uh, lih)
            except MemoryError:
                Lf, nvf = dual_norm_lower_bound(tri_mesh_fn(N), grid_index(V, N, N), 1, uh, lih)
            # ideal delta_{h,i-1}
            delta_ideal = ideal_delta(md, K, free, Gex[:i - 1], lam_ex[:i - 1])
            for auxname, aux in [("paper", AUX_PAPER), ("CR46", aux_cr[:5])]:
                lam1_low = aux[0]; lami_low = aux[i - 1]; lamip1_low = aux[i]
                lam_up_prev = lam[i - 2] if i > 1 else None
                # Theorem 5.1 constants (5.3a), Case C alpha
                terms = [(1 - lih / lamip1_low) ** -1]
                if i > 1: terms.append((lih / lam_up_prev - 1) ** -1)
                c_ih = max(terms)
                gap_ok = (lih < lamip1_low) and (i == 1 or lam_up_prev < lih)
                alpha = np.sqrt(2.0) * c_ih * CI * CS * h * eta_res
                chi = Uchi[:, i - 1]; chi_norm = np.sqrt(chi @ (M @ chi)); ip_chi = uh @ (M @ chi)
                if ip_chi < 0: ip_chi = -ip_chi
                sign_ok = alpha <= ip_chi / chi_norm
                eta_i2 = eta_res ** 2 + (lih + lih) * alpha ** 2          # 2017 (5.5): (lam_ih + lam_i_up) abar^2 with lam_i_up = lam_ih
                # Theorem 5.2 (5.11)
                cbar = max((lih / lam1_low - 1) ** 2, 1.0)
                lam_up = lih
                et_orig = eta_511(lami_low, cbar, R2, lam_up, alpha ** 2)
                Tt = lam_up * alpha ** 2
                # L2 route
                Mbar = (lih - lam1_low) ** 2 / lam1_low
                et_L2 = max(-Tt + G_fun(1.0, R2 - Mbar * alpha ** 2, lami_low), 0.0)
                # new, p = i-1 (kappa = 1 needs lih <= 2 lam_{i+1}, checked with the lower bound)
                kappa1_ok = lih <= 2 * lamip1_low
                dbar = CI * CS * h * np.sqrt(lam[i - 2]) if i > 1 else 0.0
                et_new = max(-Tt + G_fun(1.0, (1 - dbar ** 2) * R2, lami_low), 0.0) if dbar < 1 else 0.0
                # new max over p
                et_new_max = 0.0; p_best = 0
                for p in range(0, i):
                    if p == i - 1:
                        kap = 1.0
                    else:
                        kap = max(1.0, (lih / aux[p] - 1) ** 2)        # aux[p] = lower bound of lam_{p+1}
                    db = CI * CS * h * np.sqrt(lam[p - 1]) if p > 0 else 0.0
                    if db >= 1: continue
                    val = max(-Tt + G_fun(kap, (1 - db ** 2) * R2, lami_low), 0.0)
                    if val > et_new_max: et_new_max, p_best = val, p
                row = dict(N=N, h=h, ndof=len(V), i=i, aux=auxname, lam_i=li, lam_ih=lih, g=g, algres=float(algres[i - 1]),
                           eta_res=eta_res, eta_res_rt2=eta_res_rt2, div_check_rt2=fl2["div_check"], R2=R2, Lf=Lf, nvf=nvf, e=e, eps=eps, ip=ip,
                           c_ih=c_ih, gap_ok=bool(gap_ok), alpha=alpha, alpha2_ge_eps=bool(alpha ** 2 >= eps), sign_ok=bool(sign_ok),
                           eta_i2=eta_i2, Iub=eta_i2 / g, cbar=cbar, lam_i_low=lami_low,
                           et_orig=et_orig, Ilb_orig=(g / et_orig if et_orig > 0 else np.inf),
                           et_L2=et_L2, Ilb_L2=(g / et_L2 if et_L2 > 0 else np.inf),
                           et_new=et_new, Ilb_new=(g / et_new if et_new > 0 else np.inf),
                           et_new_max=et_new_max, p_best=p_best, dbar=dbar, delta_ideal=delta_ideal, kappa1_ok=bool(kappa1_ok),
                           div_check=fl["div_check"], normal_jump=fl["normal_jump"], cons=fl["max_constraint_resid"],
                           # theoretical inequalities that must hold
                           chk_R2_le_eta2=bool(R2 <= eta_res ** 2 * (1 + 1e-12)),
                           chk_Lf_le_eta2=bool(Lf <= eta_res ** 2 * (1 + 1e-12)),
                           chk_ub=bool(eta_i2 >= g), chk_orig=bool(et_orig <= g), chk_L2=bool(et_L2 <= g), chk_new=bool(et_new_max <= g),
                           chk_L8=bool((1 - delta_ideal ** 2) * Lf <= g ** 2 / li + e + 1e-12),
                           chk_318b=bool(Lf <= g ** 2 / li + max((lih / lam_ex[0] - 1) ** 2, 1) * e + 1e-12))
                rows.append(row)
        print(f"N={N} done in {time.time() - t0:.1f}s; lam_h={np.round(lam[:4], 4)}")
        sys.stdout.flush()
    with open(f"{outdir}/set1_rows.json", "w") as f:
        json.dump(rows, f, indent=1, default=float)
    return rows

def table(rows, aux="paper"):
    out = ["| N | h | i | λ_i | λ_ih | λ_ih−η_i² | λ_ih−η̃²(orig) | I^lb orig | I^ub | η̃² L² | I^lb L² | η̃² new | I^lb new | p* | δ̄ | δ ideal | ᾱ² ≥ ε | (5.9) | checks |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["aux"] != aux: continue
        chk = all(r[k] for k in r if k.startswith("chk_"))
        f = lambda x: ("–" if not np.isfinite(x) else f"{x:.2f}")
        out.append(f"| {r['N']} | {r['h']:.4f} | {r['i']} | {r['lam_i']:.4f} | {r['lam_ih']:.4f} | {r['lam_ih'] - r['eta_i2']:.4f} | {r['lam_ih'] - r['et_orig']:.4f} | {f(r['Ilb_orig'])} | {r['Iub']:.2f} | {r['et_L2']:.2e} | {f(r['Ilb_L2'])} | {r['et_new']:.2e} | {f(r['Ilb_new'])} | {r['p_best']} | {r['dbar']:.3f} | {r['delta_ideal']:.3f} | {r['alpha2_ge_eps']} | {r['sign_ok']} | {'ok' if chk else 'FAIL'} |")
    return "\n".join(out)

if __name__ == "__main__":
    Nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 320
    outdir = sys.argv[2] if len(sys.argv) > 2 else "."
    Nlist = [n for n in [40, 80, 160, 320, 640] if n <= Nmax]
    rows = run(Nlist, outdir)
    for aux in ["paper", "CR46"]:
        print(f"\n### aux = {aux}\n" + table(rows, aux))
    with open(f"{outdir}/set1_table.md", "w") as f:
        for aux in ["paper", "CR46"]:
            f.write(f"\n### aux = {aux}\n" + table(rows, aux) + "\n")
