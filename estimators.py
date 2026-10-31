import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from fem_core import element_geometry, edge_structure, quad_points, assemble_p1

def rt_exponents(k):
    ex = []
    for a in range(k + 1):
        for b in range(k + 1 - a):
            ex.append((0, a, b)); ex.append((1, a, b))
    for a in range(k + 1):
        ex.append((2, a, k - a))
    return ex

def rt_eval(k, xi):
    """RT_k = [P_k]^2 + xi * P~_k on local coordinates xi (n,2). Returns Phi (n,nb,2), Div (n,nb) w.r.t. xi."""
    ex = rt_exponents(k)
    n = xi.shape[0]; nb = len(ex)
    Phi = np.zeros((n, nb, 2)); Div = np.zeros((n, nb))
    x, y = xi[:, 0], xi[:, 1]
    for j, (t, a, b) in enumerate(ex):
        mono = x**a * y**b
        dmx = a * x**(a - 1) * y**b if a > 0 else np.zeros(n)
        dmy = b * x**a * y**(b - 1) if b > 0 else np.zeros(n)
        if t == 0:
            Phi[:, j, 0] = mono; Div[:, j] = dmx
        elif t == 1:
            Phi[:, j, 1] = mono; Div[:, j] = dmy
        else:
            Phi[:, j, 0] = x * mono; Phi[:, j, 1] = y * mono
            Div[:, j] = (a + b + 2) * mono
    return Phi, Div

def pk_eval(k, xi):
    x, y = xi[:, 0], xi[:, 1]
    cols = [x**a * y**b for a in range(k + 1) for b in range(k + 1 - a)]
    return np.stack(cols, axis=1)

def p2_basis(L):
    """P2 nodal basis (3 vertex + 3 edge-opposite-vertex functions) at barycentric points L (nq,3)."""
    nq = L.shape[0]
    N2 = np.zeros((nq, 6))
    for i in range(3):
        N2[:, i] = L[:, i] * (2 * L[:, i] - 1)
        j, kk = (i + 1) % 3, (i + 2) % 3
        N2[:, 3 + i] = 4 * L[:, j] * L[:, kk]
    return N2

def p2_grad(L, G):
    """Gradients of the P2 basis: (nt,nq,6,2) from barycentric L (nq,3) and hat gradients G (nt,3,2)."""
    nt = G.shape[0]; nq = L.shape[0]
    dN2 = np.zeros((nt, nq, 6, 2))
    for i in range(3):
        j, kk = (i + 1) % 3, (i + 2) % 3
        dN2[:, :, i, :] = (4 * L[:, i] - 1)[None, :, None] * G[:, i][:, None, :]
        dN2[:, :, 3 + i, :] = 4 * (L[:, kk][None, :, None] * G[:, j][:, None, :] + L[:, j][None, :, None] * G[:, kk][:, None, :])
    return dN2

class MeshData:
    def __init__(self, V, T, bnd, deg=8):
        self.V, self.T, self.bnd = V, T, bnd
        self.nv, self.nt = len(V), len(T)
        self.area, self.G = element_geometry(V, T)
        self.E, self.TE, self.ET = edge_structure(T)
        self.bnd_edge = self.ET[:, 1] < 0
        self.X, self.W, self.L = quad_points(V, T, deg)
        self.nq = self.L.shape[0]
        P = V[T]
        self.cK = P.mean(axis=1)
        self.sK = np.max([np.linalg.norm(P[:, 1] - P[:, 0], axis=1), np.linalg.norm(P[:, 2] - P[:, 0], axis=1),
                          np.linalg.norm(P[:, 2] - P[:, 1], axis=1)], axis=0)
        self.h = self.sK.max()
        v2t = [[] for _ in range(self.nv)]
        for t in range(self.nt):
            for kk in range(3):
                v2t[T[t, kk]].append((t, kk))
        self.v2t = v2t

    def patch_key(self, a):
        """Geometry signature of the patch of vertex a (relative coordinates, local indices, boundary flags)."""
        patch = self.v2t[a]
        rel = (self.V[self.T[[t for t, _ in patch]]] - self.V[a]) / self.h
        key = (tuple(np.round(rel.ravel(), 9)), tuple(kk for _, kk in patch),
               tuple(bool(self.bnd_edge[self.TE[t, e]]) for t, _ in patch for e in range(3)), bool(self.bnd[a]))
        return key

def equilibrated_flux(md, uh, lam_h, k=2, check=True):
    """Returns dict with eta_res, sigma coefficients (nt,nb), and consistency checks."""
    V, T, G, L, W, X = md.V, md.T, md.G, md.L, md.W, md.X
    nt, nq, nv = md.nt, md.nq, md.nv
    nb = (k + 1) * (k + 3); nm = (k + 1) * (k + 2) // 2
    Xi = (X - md.cK[:, None, :]) / md.sK[:, None, None]
    Phi, Div = rt_eval(k, Xi.reshape(-1, 2))
    Phi = Phi.reshape(nt, nq, nb, 2); Div = Div.reshape(nt, nq, nb) / md.sK[:, None, None]
    Pk = pk_eval(k, Xi.reshape(-1, 2)).reshape(nt, nq, nm)
    H = np.einsum("tq,tqjd,tqld->tjl", W, Phi, Phi)
    D = np.einsum("tq,tqj,tqm->tmj", W, Div, Pk)
    Uq = np.einsum("qk,tk->tq", L, uh[T])
    Gu = np.einsum("tki,tk->ti", G, uh[T])
    # edge Gauss points, normal traces with the global edge normal
    tg, _ = np.polynomial.legendre.leggauss(k + 1); tg = 0.5 * (tg + 1); ng = k + 1
    Nrm = np.zeros((nt, 3, ng, nb))
    for e in range(3):
        g = md.TE[:, e]
        A0 = V[md.E[g, 0]]; A1 = V[md.E[g, 1]]
        d = A1 - A0; ln = np.linalg.norm(d, axis=1)
        n = np.stack([d[:, 1], -d[:, 0]], axis=1) / ln[:, None]
        for ig in range(ng):
            xi = (A0 + tg[ig] * d - md.cK) / md.sK[:, None]
            Ph, _ = rt_eval(k, xi)
            Nrm[:, e, ig, :] = np.einsum("tjd,td->tj", Ph, n)
    sigma = np.zeros((nt, nb))
    cache = {}
    max_cons = 0.0
    for a in range(nv):
        patch = md.v2t[a]
        ne_ = len(patch); n_ = ne_ * nb
        key = md.patch_key(a)
        if key not in cache:
            Hp = np.zeros((n_, n_))
            rows = []
            for i, (t, kk) in enumerate(patch):
                sl = slice(i * nb, (i + 1) * nb)
                Hp[sl, sl] = H[t]
                for m in range(nm):
                    row = np.zeros(n_); row[sl] = D[t, m]; rows.append(row)
            seen = {}
            for i, (t, kk) in enumerate(patch):
                for e in range(3):
                    seen.setdefault(md.TE[t, e], []).append((i, t, e))
            for g, lst in seen.items():
                if md.bnd_edge[g]:
                    continue
                for ig in range(ng):
                    row = np.zeros(n_)
                    for sgn, (i, t, e) in zip([1.0, -1.0], lst):
                        row[i * nb:(i + 1) * nb] = sgn * Nrm[t, e, ig]
                    rows.append(row)
            C = np.array(rows)
            Uc, Sc, Vt = np.linalg.svd(C, full_matrices=True)
            r = int((Sc > Sc.max() * 1e-10).sum())
            pinv = Vt[:r].T * (1.0 / Sc[:r]) @ Uc[:, :r].T          # (n_, nc)
            Nn = Vt[r:].T
            Hn = Nn.T @ Hp @ Nn
            cache[key] = (Hp, pinv, Nn, np.linalg.inv(Hn), C, ne_ * nm)
        Hp, pinv, Nn, Hninv, C, ndiv = cache[key]
        # data: divergence rhs and objective linear term
        d_vec = np.zeros(C.shape[0]); bp = np.zeros(n_)
        for i, (t, kk) in enumerate(patch):
            psi = L[:, kk]
            data = lam_h * Uq[t] * psi - Gu[t] @ G[t, kk]
            d_vec[i * nm:(i + 1) * nm] = np.einsum("q,qm->m", W[t] * data, Pk[t])
            bp[i * nb:(i + 1) * nb] = np.einsum("q,qjd,d->j", W[t] * psi, Phi[t], Gu[t])
        cp = pinv @ d_vec
        if check:
            max_cons = max(max_cons, np.abs(C @ cp - d_vec).max() / max(1.0, np.abs(d_vec).max()))
        z = Hninv @ (-(Nn.T @ (Hp @ cp + bp)))
        c = cp + Nn @ z
        for i, (t, kk) in enumerate(patch):
            sigma[t] += c[i * nb:(i + 1) * nb]
    # eta_res and checks
    Sq = np.einsum("tj,tqjd->tqd", sigma, Phi)                     # sigma at quad points
    eta_res2 = np.sum(W * np.sum((Gu[:, None, :] + Sq) ** 2, axis=2))
    out = {"eta_res": np.sqrt(eta_res2), "sigma": sigma, "n_patch_shapes": len(cache), "max_constraint_resid": max_cons}
    if check:
        divs = np.einsum("tj,tqj->tq", sigma, Div)
        out["div_check"] = np.abs(divs - lam_h * Uq).max() / (lam_h * np.abs(Uq).max())
        jump = 0.0
        for g in range(len(md.E)):
            if md.bnd_edge[g]:
                continue
            t1, t2 = md.ET[g]
            e1 = int(np.where(md.TE[t1] == g)[0][0]); e2 = int(np.where(md.TE[t2] == g)[0][0])
            v1 = Nrm[t1, e1] @ sigma[t1]; v2 = Nrm[t2, e2] @ sigma[t2]
            jump = max(jump, np.abs(v1 - v2).max())
        out["normal_jump"] = jump / max(1.0, np.abs(Sq).max())
    return out

def local_liftings(md, uh, lam_h):
    """Returns dict with R2 = <Res,r>^2/||grad r||^2, <Res,r>, ||grad r||^2 and patch count."""
    V, T, G, L, W = md.V, md.T, md.G, md.L, md.W
    nt, nq, nv = md.nt, md.nq, md.nv
    N2 = p2_basis(L); dN2 = p2_grad(L, G)
    A6 = np.einsum("tq,tqid,tqjd->tij", W, dN2, dN2)
    m6 = np.einsum("tq,qi->ti", W, N2)
    Uq = np.einsum("qk,tk->tq", L, uh[T])
    Gu = np.einsum("tki,tk->ti", G, uh[T])
    rq = np.zeros((nt, nq)); grq = np.zeros((nt, nq, 2))
    cache = {}
    sum_patch_energy = 0.0
    for a in range(nv):
        patch = md.v2t[a]
        key = md.patch_key(a)
        # local dof numbering: vertices then edges
        dofs = {}
        loc_ids = []
        for (t, kk) in patch:
            ids = []
            for i in range(3):
                ids.append(dofs.setdefault(("v", T[t, i]), len(dofs)))
            for i in range(3):
                ids.append(dofs.setdefault(("e", md.TE[t, i]), len(dofs)))
            loc_ids.append(ids)
        nd = len(dofs)
        if key not in cache:
            A = np.zeros((nd, nd)); m = np.zeros(nd)
            for (t, kk), ids in zip(patch, loc_ids):
                A[np.ix_(ids, ids)] += A6[t]; m[ids] += m6[t]
            if md.bnd[a]:
                keep = np.array([not ((typ == "v" and md.bnd[idx]) or (typ == "e" and md.bnd_edge[idx]))
                                 for (typ, idx) in dofs.keys()])
                Ainv = np.zeros((nd, nd))
                Ainv[np.ix_(keep, keep)] = np.linalg.inv(A[np.ix_(keep, keep)])
                cache[key] = ("bnd", Ainv)
            else:
                K = np.zeros((nd + 1, nd + 1)); K[:nd, :nd] = A; K[:nd, nd] = m; K[nd, :nd] = m
                cache[key] = ("int", np.linalg.inv(K))
        kind, Minv = cache[key]
        f = np.zeros(nd)
        for (t, kk), ids in zip(patch, loc_ids):
            psi = L[:, kk]; gpsi = G[t, kk]
            # f_i = sum_q W [ lam u psi N_i - Gu . (N_i gpsi + psi dN_i) ]
            fl = np.einsum("q,qi->i", W[t] * (lam_h * Uq[t] * psi - (Gu[t] @ gpsi)), N2) \
                 - np.einsum("q,qid,d->i", W[t] * psi, dN2[t], Gu[t])
            f[ids] += fl
        if kind == "bnd":
            r = Minv @ f
        else:
            r = (Minv @ np.concatenate([f, [0.0]]))[:nd]
        for (t, kk), ids in zip(patch, loc_ids):
            psi = L[:, kk]; gpsi = G[t, kk]
            ra = N2 @ r[ids]                       # (nq,)
            gra = np.einsum("qid,i->qd", dN2[t], r[ids])
            rq[t] += psi * ra
            grq[t] += ra[:, None] * gpsi[None, :] + psi[:, None] * gra
        sum_patch_energy += float(r @ f)                 # = ||grad r^a||^2_{omega_a}
    res_r = np.sum(W * (lam_h * Uq * rq)) - np.sum(np.einsum("tqd,td->tq", grq, Gu) * W)
    gr2 = np.sum(W * np.sum(grq ** 2, axis=2))
    return {"R2": res_r ** 2 / gr2, "res_r": res_r, "grad_r2": gr2, "sum_patch_energy": sum_patch_energy}

def analytic_at_quad(md, ufun, gradfun):
    """Normalised analytic eigenfunction and gradient at the quadrature points, plus its Rayleigh quotient."""
    U = ufun(md.X); nrm = np.sqrt(np.sum(md.W * U ** 2))
    U = U / nrm; Gr = gradfun(md.X) / nrm
    return U, Gr, np.sum(md.W * np.sum(Gr ** 2, axis=2))

def single_errors(md, uh, Uex, Gex):
    """(e, eps, ip) for one discrete eigenvector uh and a normalised analytic eigenfunction (sign fixed by ip >= 0)."""
    Uq = np.einsum("qk,tk->tq", md.L, uh[md.T]); Gu = np.einsum("tki,tk->ti", md.G, uh[md.T])
    ip = np.sum(md.W * Uq * Uex)
    s = 1.0 if ip >= 0 else -1.0
    eps = np.sum(md.W * (Uq - s * Uex) ** 2)
    e = np.sum(md.W * np.sum((Gu[:, None, :] - s * Gex) ** 2, axis=2))
    return e, eps, abs(ip)

def cluster_quantities(md, Uh, lam_h, Uex_list, lam_ex):
    """E^2 = ||A^{1/2}(g0-gh)||_S2^2, eps = ||g0-gh||_S2^2, and the overlap matrix for a cluster.
    Uh: (nv,J) discrete vectors; Uex_list: list of J normalised analytic eigenfunctions at quad points; lam_ex: exact values."""
    J = Uh.shape[1]
    Uq = np.einsum("qk,tkj->tqj", md.L, Uh[md.T])
    O = np.zeros((J, J))
    for i in range(J):
        for j in range(J):
            O[i, j] = np.sum(md.W * Uq[:, :, i] * Uex_list[j])
    eps = 2 * J - 2 * np.sum(O ** 2)
    E2 = np.sum(lam_ex) + np.sum(lam_h) - 2 * np.sum(O ** 2 * np.asarray(lam_ex)[None, :])
    return E2, eps, O

def Q_h(lam_m, lam_M, E2, eps):
    """Higher-order part of CDMSV 2020 (4.9), with eps = ||g0-gh||_S2^2 (so eps^2 = ||.||^4)."""
    return 3 * (lam_M - lam_m) ** 2 / (4 * lam_m) * eps ** 2 \
        + (3 / lam_m) * (1 + eps ** 2 / 4) * (2 * (1 + lam_M * eps / (4 * lam_m)) ** 2 * E2 ** 2 + 2 * lam_M ** 2 * eps ** 2)

def ideal_delta(md, K, free, Gex_list, lam_ex):
    """delta_{h,p} = ||(I-P_h) Pi_{<=p}||_{a->a} with P_h the Galerkin (a-orthogonal) projection onto V_h.
    Gex_list: gradients (nt,nq,2) of the normalised exact eigenfunctions u_1..u_p; lam_ex their eigenvalues."""
    p = len(Gex_list)
    if p == 0:
        return 0.0
    Kf = K[free][:, free].tocsc()
    lu = spla.splu(Kf)
    Xp = np.zeros((md.nv, p))
    for k in range(p):
        # rhs_j = (grad u_k, grad phi_j) = sum_t sum_q W grad u_k . G[t,j]
        rhs = np.zeros(md.nv)
        contrib = np.einsum("tq,tqd,tjd->tj", md.W, Gex_list[k], md.G)      # (nt,3)
        np.add.at(rhs, md.T.ravel(), contrib.ravel())
        x = np.zeros(md.nv); x[free] = lu.solve(rhs[free]); Xp[:, k] = x
    Gram = Xp.T @ (K @ Xp)                                                     # a(P_h u_k, P_h u_l)
    scale = 1.0 / np.sqrt(np.asarray(lam_ex))
    Mres = np.eye(p) - scale[:, None] * Gram * scale[None, :]                 # a((I-P_h)psi_k, (I-P_h)psi_l)
    ev = np.linalg.eigvalsh(Mres)
    return float(np.sqrt(max(ev.max(), 0.0)))

def prolong_structured(idx_c, idx_f, r):
    """Prolongation matrix from a structured grid idx_c (coarse) to idx_f (fine) refined by factor r = 2^s (same diagonal)."""
    assert r in (2, 4, 8)
    if r > 2:
        # compose: build intermediate grid of size (coarse*2)
        raise NotImplementedError
    nf = idx_f.max() + 1; nc = idx_c.max() + 1
    rows, cols, vals = [], [], []
    I, J = np.nonzero(idx_f >= 0)
    for i, j in zip(I, J):
        f = idx_f[i, j]
        if i % 2 == 0 and j % 2 == 0:
            rows.append(f); cols.append(idx_c[i // 2, j // 2]); vals.append(1.0)
        elif i % 2 == 1 and j % 2 == 0:
            for c in (idx_c[(i - 1) // 2, j // 2], idx_c[(i + 1) // 2, j // 2]):
                rows.append(f); cols.append(c); vals.append(0.5)
        elif i % 2 == 0 and j % 2 == 1:
            for c in (idx_c[i // 2, (j - 1) // 2], idx_c[i // 2, (j + 1) // 2]):
                rows.append(f); cols.append(c); vals.append(0.5)
        else:
            for c in (idx_c[(i + 1) // 2, (j - 1) // 2], idx_c[(i - 1) // 2, (j + 1) // 2]):
                rows.append(f); cols.append(c); vals.append(0.5)
    return sp.csr_matrix((vals, (rows, cols)), shape=(nf, nc))

def grid_index(V, nx, ny, a=1.0):
    """Reconstruct the (i,j) grid index array of a structured mesh from vertex coordinates."""
    idx = -np.ones((nx + 1, ny + 1), dtype=int)
    i = np.rint(V[:, 0] * nx / a).astype(int); j = np.rint(V[:, 1] * ny).astype(int)
    idx[i, j] = np.arange(len(V))
    return idx

def dual_norm_lower_bound(mesh_fn, idx_c, levels, uh, lam_h):
    """Lower bound ||grad w_f||^2 <= ||Res||_{-1}^2 by Galerkin on a nested mesh refined 'levels' times by 2.
    mesh_fn(s) must return (V, T, bnd, idx) for refinement step s (s=0 is the coarse mesh)."""
    u = uh.copy(); idx_prev = idx_c
    for s in range(1, levels + 1):
        V, T, bnd, idx = mesh_fn(s)
        P = prolong_structured(idx_prev, idx, 2)
        u = P @ u; idx_prev = idx
    K, M, _, _ = assemble_p1(V, T)
    free = ~bnd
    res = lam_h * (M @ u) - K @ u
    Kf = K[free][:, free].tocsc()
    w = spla.spsolve(Kf, res[free])
    return float(res[free] @ w), len(V)
