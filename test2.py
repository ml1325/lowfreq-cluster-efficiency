# Test 2: the cluster {lambda_10, lambda_11, lambda_12} on the rectangle (0,1.1) x (0,1), Section 5 and Table 1.
import sys
import numpy as np
import scipy.sparse.linalg as spla
from fem import mesh_rect, assemble_p1, eig_p1, cr_lower_bounds, prolong, rect_modes
from estimators import Mesh, equilibrated_flux, residual_lifting, cluster_errors, Q_h, angle_matrix, angle
from test1 import write_csv

A = 1.1
CI, CS = 0.493 / np.sqrt(2), 1.0
P = 9  # cluster indices m..M = 10..12, p = m - 1


def lv_lower(V, T, U, lam, idx, levels, Nx, Ny):
    # Gram matrix of Galerkin liftings of the residuals on a mesh refined `levels` times
    for s in range(1, levels + 1):
        V, T, bnd, idx_f = mesh_rect(A, Nx * 2**s, Ny * 2**s)
        U, idx = prolong(idx, idx_f) @ U, idx_f
    K, M = assemble_p1(V, T)
    free = ~bnd
    R = ((M @ U) * lam - K @ U)[free]
    G = R.T @ spla.splu(K[free][:, free].tocsc()).solve(R)
    s = lam**-1.5
    return np.linalg.eigvalsh(s[:, None] * (G + G.T) / 2 * s[None, :]).max()


def run(Nys):
    modes = rect_modes(A, 12)
    lam_ex = np.array([mode[0] for mode in modes])
    # guaranteed lower bounds of lambda_1..14 from the coarsest mesh; aux[k] <= lambda_{k+1}
    V, T, _, _ = mesh_rect(A, 44, 40)
    aux = cr_lower_bounds(V, T, 14)
    rows, rows_p = [], []
    for Ny in Nys:
        Nx = round(1.1 * Ny)
        V, T, bnd, idx = mesh_rect(A, Nx, Ny)
        free = ~bnd
        K, M = assemble_p1(V, T)
        lam, U, res = eig_p1(K, M, free, 14, 0.0)
        m8, m4 = Mesh(V, T, bnd, 8), Mesh(V, T, bnd, 4)
        h = m8.h
        Uex, Gex = zip(*[m8.exact(u, g) for _, u, g in modes])
        flux = [equilibrated_flux(m4, U[:, k], lam[k]) for k in range(12)]
        eta2 = np.array([f[0]**2 for f in flux])

        # Table 1
        cl = slice(P, 12)
        E2, eps, O = cluster_errors(m8, U[:, cl], lam[cl], Uex[cl], lam_ex[cl])
        Qh = Q_h(lam_ex[P], lam_ex[11], E2, eps)
        R2 = sum(residual_lifting(m8, U[:, k], lam[k]) for k in range(P, 12))
        lam_mh, lam_Mh = lam[P], lam[11]
        gap_ok = lam[P - 1] < lam_mh and lam_Mh < aux[12]
        cbar = max((lam_Mh / aux[0] - 1)**2, 1)                                  # Eq. (1)
        c_h = max((lam_mh / lam[P - 1] - 1)**-1, (1 - lam_Mh / aux[12])**-1)
        ell = c_h**2 * (lam_Mh - aux[0])**2 * (CI * CS * h)**2 / aux[0]          # Proposition 1
        kappa = max(1, (lam_Mh / aux[12] - 1)**2)                                # Eq. (7), C_{m-1} = kappa_+
        dbar = CI * CS * h * np.sqrt(lam[P - 1])                                 # Eq. (9)
        D = angle_matrix(m8, K, free, Gex[:P], lam_ex[:P])

        # Theorem 1(iv) for p = 1..9 and the first inequality in its proof
        Ol = cluster_errors(m8, U[:, :P], lam[:P], Uex[:P], lam_ex[:P])[2]
        for p in range(1, P + 1):
            low = aux[p]
            dhat = np.sqrt(np.sum((1 - lam[:p] / low)**-2 * eta2[:p] / lam[:p])) if lam[p - 1] < low else np.inf
            C = np.sqrt(lam_ex[None, :p] / lam[:p, None]) * Ol[:p, :p]
            sin_theta = np.sqrt(max(1 - np.linalg.svd(C, compute_uv=False).min()**2, 0))
            Cp = max(kappa, (lam_Mh / low - 1)**2) if p < P else kappa
            rows_p.append(dict(Ny=Ny, p=p, dhat=dhat, sin_theta=sin_theta, delta=angle(D, p), C_p=Cp,
                               const=Cp / (1 - dhat**2) if dhat < 1 else np.inf))
        best = min((r for r in rows_p if r["Ny"] == Ny), key=lambda r: r["const"])

        # single-gap bound of Liu and Vejchodsky (2022) for p = 9: upper end from the fluxes,
        # lower end from liftings of the residuals on a finer mesh
        gap = 1 / lam[P - 1] - 1 / aux[P]
        q = np.array([(m4.grad(U[:, k])[:, None] + flux[k][1]) / lam[k]**1.5 for k in range(P)])
        lv_up = np.sqrt(np.linalg.eigvalsh(np.einsum("tq,ktqd,ltqd->kl", m4.W, q, q)).max()) / gap
        lv_low = np.sqrt(lv_lower(V, T, U[:, :P], lam[:P], idx, 2 if Ny <= 160 else 1, Nx, Ny)) / gap

        den = E2 + Qh
        sv = np.linalg.svd(O, compute_uv=False)
        rows.append(dict(Ny=Ny, vertices=len(V), h=h, E2=E2, Qh_E2=Qh / E2,
                         ratio_low=R2 / den, ratio_up=eta2[cl].sum() / den,
                         cbar=cbar, const_orig=(cbar * E2 + Qh) / den,
                         ell=ell, const_L2=1 / (1 - ell) if ell < 1 else np.inf,
                         dbar=dbar, delta=angle(D, P), hs=np.sqrt(np.trace(D)), const_dbar=kappa / (1 - dbar**2),
                         dhat=rows_p[-1]["dhat"], const_dhat=rows_p[-1]["const"], best_p=best["p"], const_best=best["const"],
                         lv_low=lv_low, lv_up=lv_up, sigma_min=sv.min(), kappa=kappa, gap_ok=gap_ok, res=res[:12].max()))
        r = rows[-1]
        print(f"Ny={Ny:4d}  ratio [{r['ratio_low']:.4f}, {r['ratio_up']:.4f}]  orig {r['const_orig']:.2f}  "
              f"L2 {r['const_L2']:.4f}  dbar {r['const_dbar']:.5f}  dhat {r['const_dhat']:.4f}  "
              f"best {r['const_best']:.4f} (p={r['best_p']})  LV [{lv_low:.4f}, {lv_up:.4f}]", flush=True)
    return rows, rows_p


if __name__ == "__main__":
    Nymax = int(sys.argv[1]) if len(sys.argv) > 1 else 320
    out = sys.argv[2] if len(sys.argv) > 2 else "results"
    rows, rows_p = run([N for N in (40, 80, 160, 320) if N <= Nymax])
    write_csv(f"{out}/test2.csv", rows)
    write_csv(f"{out}/test2_p.csv", rows_p)
