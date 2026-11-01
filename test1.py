# Test 1: single eigenvalues on the triangle (0,0), (1,0), (0,1), Section 5 and Fig. 1.
import sys
import numpy as np
from fem import mesh_triangle, assemble_p1, eig_p1, prolong, triangle_modes
from estimators import Mesh, equilibrated_flux, residual_lifting, angle_matrix, angle

CI, CS = 0.493 / np.sqrt(2), 1.0
AUX = [49.2883, 98.4296, 127.937, 166.975, 196.439]  # lower bounds of lambda_1..5 used by Cances et al. (2017)


def eta_tilde(kappa, S, lam, shift):
    # max{-shift + (sqrt(lam^2 kappa^2 + 4 lam S) - lam kappa) / 2, 0} as in Eqs. (12) and (13)
    root = 2 * S / (kappa + np.sqrt(kappa**2 + 4 * S / lam)) if S > 0 else 0.0
    return max(root - shift, 0.0)


def write_csv(path, rows):
    with open(path, "w") as f:
        f.write(",".join(rows[0]) + "\n")
        for r in rows:
            f.write(",".join(f"{v:.10g}" if isinstance(v, float) else str(v) for v in r.values()) + "\n")


def run(Ns):
    modes = triangle_modes(4)
    lam_ex = np.array([mode[0] for mode in modes])
    # chi_i in the sign condition: discrete eigenfunctions for N = 20, interpolated to the finer meshes
    V, T, bnd, idx_chi = mesh_triangle(20)
    _, chi, _ = eig_p1(*assemble_p1(V, T), ~bnd, 6, 40.0)
    rows = []
    for N in Ns:
        V, T, bnd, idx = mesh_triangle(N)
        free = ~bnd
        K, M = assemble_p1(V, T)
        lam, U, res = eig_p1(K, M, free, 6, 40.0)
        while len(chi) < len(V):
            n = 2 * (idx_chi.shape[0] - 1)
            idx_f = mesh_triangle(n)[3]
            chi, idx_chi = prolong(idx_chi, idx_f) @ chi, idx_f
        m8, m4 = Mesh(V, T, bnd, 8), Mesh(V, T, bnd, 4)
        h = m8.h
        Gex = [m8.exact(u, g)[1] for _, u, g in modes]
        eta = [equilibrated_flux(m4, U[:, k], lam[k])[0] for k in range(4)]
        D = angle_matrix(m8, K, free, Gex[:3], lam_ex[:3])
        for i in range(1, 5):
            lih, li, low = lam[i - 1], lam_ex[i - 1], AUX[i - 1]
            err = lih - li
            R2 = residual_lifting(m8, U[:, i - 1], lih)
            # Case C of Cances et al. (2017)
            c = (1 - lih / AUX[i])**-1 if i == 1 else max((1 - lih / AUX[i])**-1, (lih / lam[i - 2] - 1)**-1)
            gap_ok = lih < AUX[i] and (i == 1 or lam[i - 2] < lih)
            alpha = np.sqrt(2) * c * CI * CS * h * eta[i - 1]
            x = chi[:, i - 1]
            sign_ok = alpha <= abs(U[:, i - 1] @ (M @ x)) / np.sqrt(x @ (M @ x))
            ub = eta[i - 1]**2 + 2 * lih * alpha**2
            Tt = lih * alpha**2
            cbar = max((lih / AUX[0] - 1)**2, 1)
            orig = eta_tilde(cbar, R2, low, Tt)                                         # Eq. (12)
            L2 = eta_tilde(1, R2 - (lih - AUX[0])**2 / AUX[0] * alpha**2, low, Tt)      # Proposition 1
            # Corollary 1 with p = i - 1, using Eq. (9) and Eq. (10)
            p = i - 1
            dbar = CI * CS * h * np.sqrt(lam[p - 1]) if p else 0.0
            dhat = np.sqrt(np.sum((1 - lam[:p] / low)**-2 * np.array(eta[:p])**2 / lam[:p]))
            new_bar = eta_tilde(1, (1 - dbar**2) * R2, low, Tt) if dbar < 1 else 0.0
            new_hat = eta_tilde(1, (1 - dhat**2) * R2, low, Tt) if (p == 0 or lam[p - 1] < low) and dhat < 1 else 0.0
            delta = angle(D, p)
            index = lambda e: err / e if e > 0 else np.inf
            rows.append(dict(N=N, h=h, i=i, lam=li, lam_h=lih, cbar=cbar, I_ub=ub / err, ub_orig=lih - orig,
                             I_orig=index(orig), I_L2=index(L2), dbar=dbar, dhat=dhat, delta=delta,
                             I_dbar=index(new_bar), I_dhat=index(new_hat),
                             gap_ok=gap_ok, sign_ok=bool(sign_ok), res=res[i - 1]))
            r = rows[-1]
            print(f"N={N:4d} i={i}  I_ub={r['I_ub']:.2f}  I_lb: orig {r['I_orig']:.2f}  L2 {r['I_L2']:.2f}  "
                  f"dbar {r['I_dbar']:.2f}  dhat {r['I_dhat']:.2f}  delta/dbar={delta / dbar if dbar else 0:.3f}",
                  flush=True)
    return rows


if __name__ == "__main__":
    Nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 320
    rows = run([N for N in (40, 80, 160, 320) if N <= Nmax])
    write_csv(sys.argv[2] if len(sys.argv) > 2 else "results/test1.csv", rows)
