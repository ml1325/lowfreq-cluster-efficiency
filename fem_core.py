import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def mesh_triangle(N):
    """Structured mesh of the unit right triangle with legs on the axes. Returns (V, T, bnd)."""
    idx = -np.ones((N + 1, N + 1), dtype=int)
    V = []
    for j in range(N + 1):
        for i in range(N + 1 - j):
            idx[i, j] = len(V)
            V.append((i / N, j / N))
    V = np.array(V)
    T = []
    for j in range(N):
        for i in range(N - j):
            T.append((idx[i, j], idx[i + 1, j], idx[i, j + 1]))            # lower-left triangle
            if i + j + 2 <= N:
                T.append((idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]))  # upper-right triangle
    T = np.array(T, dtype=int)
    x, y = V[:, 0], V[:, 1]
    bnd = (np.abs(x) < 1e-12) | (np.abs(y) < 1e-12) | (np.abs(x + y - 1) < 1e-12)
    return V, T, bnd

def mesh_rect(a, Nx, Ny):
    """Structured mesh of (0,a)x(0,1). Returns (V, T, bnd)."""
    idx = np.arange((Nx + 1) * (Ny + 1)).reshape(Nx + 1, Ny + 1)
    xs = np.linspace(0, a, Nx + 1); ys = np.linspace(0, 1, Ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    V = np.column_stack([X.ravel(), Y.ravel()])
    T = []
    for i in range(Nx):
        for j in range(Ny):
            T.append((idx[i, j], idx[i + 1, j], idx[i, j + 1]))
            T.append((idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]))
    T = np.array(T, dtype=int)
    x, y = V[:, 0], V[:, 1]
    bnd = (np.abs(x) < 1e-12) | (np.abs(y) < 1e-12) | (np.abs(x - a) < 1e-12) | (np.abs(y - 1) < 1e-12)
    return V, T, bnd

def element_geometry(V, T):
    """Areas (positive) and constant gradients of the P1 hat functions: G[t, k, :] = grad of hat at vertex k."""
    P = V[T]                                       # (nt, 3, 2)
    d1 = P[:, 1] - P[:, 0]; d2 = P[:, 2] - P[:, 0]
    det = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    assert np.all(det > 0), "triangles must be counter-clockwise"
    area = 0.5 * det
    # grad lambda_k = rot of opposite edge / (2 area)
    G = np.empty((len(T), 3, 2))
    for k in range(3):
        a_, b_ = P[:, (k + 1) % 3], P[:, (k + 2) % 3]
        G[:, k, 0] = (a_[:, 1] - b_[:, 1]) / det
        G[:, k, 1] = (b_[:, 0] - a_[:, 0]) / det
    return area, G

def edge_structure(T):
    """Unique edges. Returns E (ne,2) with v0<v1, TE (nt,3) edge index opposite vertex k, and ET (ne,2) triangles (-1 if boundary)."""
    nt = len(T)
    loc = np.stack([T[:, [1, 2]], T[:, [2, 0]], T[:, [0, 1]]], axis=1)   # edge opposite vertex k
    loc_sorted = np.sort(loc, axis=2).reshape(-1, 2)
    E, inv = np.unique(loc_sorted, axis=0, return_inverse=True)
    TE = inv.reshape(nt, 3)
    ET = -np.ones((len(E), 2), dtype=int)
    for t in range(nt):
        for k in range(3):
            e = TE[t, k]
            ET[e, 1 if ET[e, 0] >= 0 else 0] = t
    return E, TE, ET

def assemble_p1(V, T):
    area, G = element_geometry(V, T)
    nt = len(T); nv = len(V)
    Kloc = np.einsum("tki,tli->tkl", G, G) * area[:, None, None]
    Mloc = (np.ones((3, 3)) + np.eye(3)) / 12.0 * area[:, None, None]
    rows = np.repeat(T, 3, axis=1).ravel(); cols = np.tile(T, (1, 3)).ravel()
    K = sp.csr_matrix((Kloc.ravel(), (rows, cols)), shape=(nv, nv))
    M = sp.csr_matrix((Mloc.ravel(), (rows, cols)), shape=(nv, nv))
    return K, M, area, G

def eig_p1(K, M, free, k, sigma):
    """k eigenpairs of K u = lam M u on free dofs, shift-invert around sigma. Returns lam (sorted), U (nv x k, zero on Dirichlet nodes), and algebraic residuals."""
    Kf = K[free][:, free].tocsc(); Mf = M[free][:, free].tocsc()
    lam, X = spla.eigsh(Kf, k=k, M=Mf, sigma=sigma, which="LM", tol=1e-12)
    order = np.argsort(lam); lam = lam[order]; X = X[:, order]
    # M-normalise and report algebraic residual ||K x - lam M x||_{M^{-1}}-free proxy: 2-norm residual / ||x||
    res = np.empty(k)
    for j in range(k):
        x = X[:, j]
        nrm = np.sqrt(x @ (Mf @ x)); x /= nrm; X[:, j] = x
        res[j] = np.linalg.norm(Kf @ x - lam[j] * (Mf @ x))
    U = np.zeros((K.shape[0], k)); U[free] = X
    return lam, U, res

def assemble_cr(V, T):
    """Crouzeix-Raviart: dof on edges, basis phi_e = 1 - 2 lambda_{opposite vertex}. Returns K, M, E, TE."""
    area, G = element_geometry(V, T)
    E, TE, ET = edge_structure(T)
    nt = len(T); ne = len(E)
    Gcr = -2.0 * G                                   # grad phi_{e opposite k} = -2 grad lambda_k
    Kloc = np.einsum("tki,tli->tkl", Gcr, Gcr) * area[:, None, None]
    Mloc = np.eye(3)[None] * (area / 3.0)[:, None, None]   # int_T phi_i phi_j = |T|/3 delta_ij
    rows = np.repeat(TE, 3, axis=1).ravel(); cols = np.tile(TE, (1, 3)).ravel()
    K = sp.csr_matrix((Kloc.ravel(), (rows, cols)), shape=(ne, ne))
    M = sp.csr_matrix((Mloc.ravel(), (rows, cols)), shape=(ne, ne))
    bnd_edge = ET[:, 1] < 0
    return K, M, E, TE, bnd_edge

def cr_lower_bounds(V, T, k, kappa=0.1893):
    """Guaranteed lower bounds lam_k >= lam_CR,k / (1 + kappa^2 H^2 lam_CR,k) (Carstensen-Gedicke 2014, Thm 3.1), H = max diameter."""
    K, M, E, TE, bnd_edge = assemble_cr(V, T)
    free = ~bnd_edge
    Kf = K[free][:, free].tocsc(); Mf = M[free][:, free].tocsc()
    lam, _ = spla.eigsh(Kf, k=k, M=Mf, sigma=0.0, which="LM", tol=1e-12)
    lam = np.sort(lam)
    P = V[T]
    H = max(np.linalg.norm(P[:, 1] - P[:, 0], axis=1).max(), np.linalg.norm(P[:, 2] - P[:, 0], axis=1).max(),
            np.linalg.norm(P[:, 2] - P[:, 1], axis=1).max())
    return lam / (1.0 + kappa**2 * H**2 * lam), lam, H

def dunavant(degree=8):
    """Dunavant rule on the reference triangle (barycentric weights normalised to 1). degree 8: 16 points."""
    if degree <= 4:
        # 6-point degree-4
        a1, a2 = 0.445948490915965, 0.091576213509771
        w1, w2 = 0.223381589678011, 0.109951743655322
        pts = [(a1, a1, 1 - 2 * a1), (a1, 1 - 2 * a1, a1), (1 - 2 * a1, a1, a1),
               (a2, a2, 1 - 2 * a2), (a2, 1 - 2 * a2, a2), (1 - 2 * a2, a2, a2)]
        w = [w1] * 3 + [w2] * 3
    else:
        # 16-point degree-8 (Dunavant 1985)
        data = [
            (0.333333333333333, 0.333333333333333, 0.144315607677787, 1),
            (0.081414823414554, 0.459292588292723, 0.095091634267285, 3),
            (0.658861384496480, 0.170569307751760, 0.103217370534718, 3),
            (0.898905543365938, 0.050547228317031, 0.032458497623198, 3),
            (0.008394777409958, 0.263112829634638, 0.027230314174435, 6),
        ]
        pts, w = [], []
        for a, b, wt, mult in data:
            if mult == 1:
                pts.append((a, a, a)); w.append(wt)
            elif mult == 3:
                c = 1 - 2 * b
                for p in [(c, b, b), (b, c, b), (b, b, c)]:
                    pts.append(p); w.append(wt)
            else:
                c = 1 - a - b
                for p in [(a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)]:
                    pts.append(p); w.append(wt)
    return np.array(pts), np.array(w)

def quad_points(V, T, degree=8):
    """Physical quadrature points X[t,q,:], weights W[t,q] (including area), barycentric L[q,3]."""
    L, w = dunavant(degree)
    P = V[T]                                         # (nt,3,2)
    X = np.einsum("qk,tki->tqi", L, P)
    area, _ = element_geometry(V, T)
    W = area[:, None] * w[None, :]
    return X, W, L

def analytic_triangle(nmax=8):
    out = []
    for m in range(2, nmax + 1):
        for n in range(1, m):
            out.append((np.pi**2 * (m * m + n * n), m, n))
    out.sort()
    return out

def u_triangle(m, n, X):
    x, y = X[..., 0], X[..., 1]
    return np.sin(m * np.pi * (1 - y)) * np.sin(n * np.pi * x) - np.sin(n * np.pi * (1 - y)) * np.sin(m * np.pi * x)

def grad_u_triangle(m, n, X):
    x, y = X[..., 0], X[..., 1]
    ux = np.sin(m * np.pi * (1 - y)) * n * np.pi * np.cos(n * np.pi * x) - np.sin(n * np.pi * (1 - y)) * m * np.pi * np.cos(m * np.pi * x)
    uy = -m * np.pi * np.cos(m * np.pi * (1 - y)) * np.sin(n * np.pi * x) + n * np.pi * np.cos(n * np.pi * (1 - y)) * np.sin(m * np.pi * x)
    return np.stack([ux, uy], axis=-1)

def analytic_rect(a, nmax=8):
    out = []
    for m in range(1, nmax + 1):
        for n in range(1, nmax + 1):
            out.append((np.pi**2 * (m * m / a**2 + n * n), m, n))
    out.sort()
    return out

def u_rect(a, m, n, X):
    return np.sin(m * np.pi * X[..., 0] / a) * np.sin(n * np.pi * X[..., 1])

def grad_u_rect(a, m, n, X):
    x, y = X[..., 0], X[..., 1]
    return np.stack([m * np.pi / a * np.cos(m * np.pi * x / a) * np.sin(n * np.pi * y),
                     n * np.pi * np.sin(m * np.pi * x / a) * np.cos(n * np.pi * y)], axis=-1)

def normalise_analytic(ufun, X, W):
    """L2 norm of an analytic function by quadrature (for normalising exact eigenfunctions)."""
    return np.sqrt(np.sum(W * ufun(X) ** 2))

def p1_at_quad(U, T, L):
    """Values of P1 function(s) U (nv or nv x k) at quadrature points: (nt, nq[, k])."""
    return np.einsum("qk,tk...->tq...", L, U[T])

def p1_grad(U, T, G):
    """Constant gradients per element of P1 function(s): (nt, 2[, k])."""
    return np.einsum("tki,tk...->ti...", G, U[T])

if __name__ == "__main__":
    # self-test 1: triangle eigenvalues vs analytic; CR lower bounds below analytic
    V, T, bnd = mesh_triangle(40)
    K, M, area, G = assemble_p1(V, T)
    free = ~bnd
    lam, U, res = eig_p1(K, M, free, 6, sigma=40.0)
    ana = analytic_triangle()
    print("P1 triangle N=40: lam_h =", np.round(lam, 4))
    print("analytic          =", np.round([a[0] for a in ana[:6]], 4))
    print("alg residuals:", res.max())
    lb, lcr, H = cr_lower_bounds(V, T, 6)
    print("CR lower bounds   =", np.round(lb, 4), " (H=%.4f)" % H, " all below analytic:", np.all(lb <= [a[0] for a in ana[:6]]))
    # analytic eigenfunction vanishes on boundary and reproduces lambda via Rayleigh quotient
    X, W, L = quad_points(V, T)
    for lam_ex, m, n in ana[:4]:
        u = u_triangle(m, n, X); gu = grad_u_triangle(m, n, X)
        rq = np.sum(W * np.sum(gu**2, -1)) / np.sum(W * u**2)
        print(f"  (m,n)=({m},{n}) lambda={lam_ex:.4f} Rayleigh quotient by quadrature={rq:.4f}  max|u| on bnd={np.abs(u_triangle(m,n,V[bnd])).max():.1e}")
    # self-test 2: rectangle
    a = 1.1; V, T, bnd = mesh_rect(a, 44, 40)
    K, M, area, G = assemble_p1(V, T)
    lam, U, res = eig_p1(K, M, ~bnd, 13, sigma=100.0)
    ana = analytic_rect(a)
    print("P1 rect N=40: lam_h =", np.round(lam, 3))
    print("analytic          =", np.round([x[0] for x in ana[:13]], 3))
    lb, lcr, H = cr_lower_bounds(V, T, 13)
    print("CR lower bounds   =", np.round(lb, 3), " all below:", np.all(lb <= [x[0] for x in ana[:13]]))
