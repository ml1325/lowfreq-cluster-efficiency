import itertools
import numpy as np
from estimators import MeshData


def bary_monomials(k):
    return [(a, b, k - a - b) for a in range(k + 1) for b in range(k + 1 - a)]


def lagrange_nodes(k):
    """Nodal barycentric points of P_k: vertices, then (k-1) nodes per edge opposite vertex i (ordered from
    vertex j=(i+1)%3 towards vertex kk=(i+2)%3), then interior nodes. Returns list of (L, tag)."""
    nodes = []
    for i in range(3):
        L = np.zeros(3); L[i] = 1.0; nodes.append((L, ("v", i)))
    for i in range(3):
        j, kk = (i + 1) % 3, (i + 2) % 3
        for s in range(1, k):
            L = np.zeros(3); L[j] = (k - s) / k; L[kk] = s / k
            nodes.append((L, ("e", i, s)))          # s-th node from vertex j towards vertex kk
    for a in range(1, k):
        for b in range(1, k - a):
            c = k - a - b
            if c >= 1:
                nodes.append((np.array([a, b, c]) / k, ("i", a, b)))
    return nodes


def pk_nodal_basis(k, L, G):
    """Values N (nq, nd) and gradients dN (nt, nq, nd, 2) of the P_k nodal basis at barycentric points L (nq,3),
    with hat-function gradients G (nt,3,2)."""
    mons = bary_monomials(k)
    nodes = lagrange_nodes(k)
    nd = len(nodes); assert nd == len(mons) == (k + 1) * (k + 2) // 2
    def mono_val(Lp):
        return np.stack([Lp[:, 0] ** a * Lp[:, 1] ** b * Lp[:, 2] ** c for (a, b, c) in mons], axis=1)
    Vn = mono_val(np.array([n[0] for n in nodes]))        # (nd, nd)
    C = np.linalg.inv(Vn)                                  # basis_j = sum_m C[m, j] mono_m
    N = mono_val(L) @ C                                    # (nq, nd)
    # derivatives of monomials w.r.t. barycentric coordinates
    dmono = np.zeros((L.shape[0], len(mons), 3))
    for mi, (a, b, c) in enumerate(mons):
        if a > 0: dmono[:, mi, 0] = a * L[:, 0] ** (a - 1) * L[:, 1] ** b * L[:, 2] ** c
        if b > 0: dmono[:, mi, 1] = b * L[:, 0] ** a * L[:, 1] ** (b - 1) * L[:, 2] ** c
        if c > 0: dmono[:, mi, 2] = c * L[:, 0] ** a * L[:, 1] ** b * L[:, 2] ** (c - 1)
    dNbary = np.einsum("qmi,mj->qji", dmono, C)            # (nq, nd, 3)
    dN = np.einsum("qji,tid->tqjd", dNbary, G)             # (nt, nq, nd, 2)
    return N, dN, nodes


def local_liftings_pk(md, uh, lam_h, k=3):
    """P_k version of estimators.local_liftings. Returns dict with R2, res_r, grad_r2."""
    V, T, G, L, W = md.V, md.T, md.G, md.L, md.W
    nt, nq, nv = md.nt, md.nq, md.nv
    N, dN, nodes = pk_nodal_basis(k, L, G)
    nd_el = len(nodes)
    A_el = np.einsum("tq,tqid,tqjd->tij", W, dN, dN)
    m_el = np.einsum("tq,qi->ti", W, N)
    Uq = np.einsum("qk,tk->tq", L, uh[T])
    Gu = np.einsum("tki,tk->ti", G, uh[T])
    rq = np.zeros((nt, nq)); grq = np.zeros((nt, nq, 2))
    cache = {}
    for a in range(nv):
        patch = md.v2t[a]
        key = md.patch_key(a)
        dofs = {}; loc_ids = []
        for (t, kk) in patch:
            ids = []
            for (Lp, tag) in nodes:
                if tag[0] == "v":
                    ids.append(dofs.setdefault(("v", T[t, tag[1]]), len(dofs)))
                elif tag[0] == "e":
                    i, s = tag[1], tag[2]
                    j, k2 = (i + 1) % 3, (i + 2) % 3
                    g = md.TE[t, i]
                    # orientation-free identity: node at distance s/k from vertex T[t,j] towards T[t,k2]
                    vj, vk = T[t, j], T[t, k2]
                    ids.append(dofs.setdefault(("e", g, vj, vk, s) if vj < vk else ("e", g, vk, vj, k - s), len(dofs)))
                else:
                    ids.append(dofs.setdefault(("i", t, tag[1], tag[2]), len(dofs)))
            loc_ids.append(ids)
        nd = len(dofs)
        if key not in cache:
            A = np.zeros((nd, nd)); m = np.zeros(nd)
            for (t, kk), ids in zip(patch, loc_ids):
                A[np.ix_(ids, ids)] += A_el[t]; m[ids] += m_el[t]
            if md.bnd[a]:
                keep = np.ones(nd, dtype=bool)
                for key_d, idx in dofs.items():
                    if key_d[0] == "v" and md.bnd[key_d[1]]:
                        keep[idx] = False
                    if key_d[0] == "e" and md.bnd_edge[key_d[1]]:
                        keep[idx] = False
                Ainv = np.zeros((nd, nd))
                Ainv[np.ix_(keep, keep)] = np.linalg.inv(A[np.ix_(keep, keep)])
                cache[key] = ("bnd", Ainv)
            else:
                Kk = np.zeros((nd + 1, nd + 1)); Kk[:nd, :nd] = A; Kk[:nd, nd] = m; Kk[nd, :nd] = m
                cache[key] = ("int", np.linalg.inv(Kk))
        kind, Minv = cache[key]
        f = np.zeros(nd)
        for (t, kk), ids in zip(patch, loc_ids):
            psi = L[:, kk]; gpsi = G[t, kk]
            fl = np.einsum("q,qi->i", W[t] * (lam_h * Uq[t] * psi - (Gu[t] @ gpsi)), N) \
                 - np.einsum("q,qid,d->i", W[t] * psi, dN[t], Gu[t])
            f[ids] += fl
        r = Minv @ f if kind == "bnd" else (Minv @ np.concatenate([f, [0.0]]))[:nd]
        for (t, kk), ids in zip(patch, loc_ids):
            psi = L[:, kk]; gpsi = G[t, kk]
            ra = N @ r[ids]
            gra = np.einsum("qid,i->qd", dN[t], r[ids])
            rq[t] += psi * ra
            grq[t] += ra[:, None] * gpsi[None, :] + psi[:, None] * gra
    res_r = np.sum(W * (lam_h * Uq * rq)) - np.sum(np.einsum("tqd,td->tq", grq, Gu) * W)
    gr2 = np.sum(W * np.sum(grq ** 2, axis=2))
    return {"R2": res_r ** 2 / gr2, "res_r": res_r, "grad_r2": gr2, "k": k}


def collapsed_gauss(n):
    """Collapsed (Duffy) Gauss-Legendre rule on the reference triangle: barycentric points (nq,3), weights summing to 1."""
    x, w = np.polynomial.legendre.leggauss(n); x = 0.5 * (x + 1); w = 0.5 * w
    pts, wts = [], []
    for i in range(n):
        for j in range(n):
            xi = x[i]; eta = x[j] * (1 - x[i])
            pts.append((1 - xi - eta, xi, eta)); wts.append(w[i] * w[j] * (1 - x[i]))
    pts = np.array(pts); wts = np.array(wts); wts /= wts.sum()
    return pts, wts


def with_gauss_quadrature(md, n):
    """Return a copy of md whose quadrature is the collapsed Gauss rule with n^2 points."""
    import copy
    md2 = copy.copy(md)
    L, w = collapsed_gauss(n)
    P = md.V[md.T]
    md2.X = np.einsum("qk,tki->tqi", L, P)
    md2.W = md.area[:, None] * w[None, :]
    md2.L = L; md2.nq = L.shape[0]
    return md2


if __name__ == "__main__":
    # self-test: the P2 version must reproduce estimators.local_liftings
    from fem_core import *
    from estimators import local_liftings, equilibrated_flux
    V, T, bnd = mesh_triangle(20); md = MeshData(V, T, bnd, deg=8)
    K, M, area, G = assemble_p1(V, T); lam, U, res = eig_p1(K, M, ~bnd, 4, 40.0)
    for i in [1, 2]:
        uh = U[:, i - 1]; lh = lam[i - 1]
        a = local_liftings(md, uh, lh)["R2"]; b = local_liftings_pk(md, uh, lh, k=2)["R2"]; c = local_liftings_pk(md, uh, lh, k=3)["R2"]
        eta2 = equilibrated_flux(md, uh, lh, k=1)["eta_res"] ** 2
        mdg = with_gauss_quadrature(md, 10)
        d = local_liftings_pk(mdg, uh, lh, k=3)["R2"]
        print(f"i={i}: P2 orig {a:.10e}  P2 new {b:.10e}  P3 {c:.10e}  P3 gauss10 {d:.10e}  eta_res^2 {eta2:.6e}  R2<=eta2: {c <= eta2}")
