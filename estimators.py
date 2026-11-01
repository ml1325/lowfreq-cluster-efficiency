import numpy as np
import scipy.sparse.linalg as spla
from fem import geometry, edges, dunavant


class Mesh:
    def __init__(self, V, T, bnd, deg=8):
        self.V, self.T, self.bnd = V, T, bnd
        self.nv, self.nt = len(V), len(T)
        self.area, self.G = geometry(V, T)
        self.E, self.TE, self.bnd_edge = edges(T)
        self.L, w = dunavant(deg)
        self.X = np.einsum("qk,tki->tqi", self.L, V[T])
        self.W = self.area[:, None] * w
        self.nq = len(w)
        P = V[T]
        self.cK = P.mean(axis=1)
        self.sK = np.max([np.linalg.norm(P[:, i] - P[:, j], axis=1) for i, j in [(1, 0), (2, 0), (2, 1)]], axis=0)
        self.h = self.sK.max()
        self.v2t = [[] for _ in range(self.nv)]
        for t in range(self.nt):
            for k in range(3):
                self.v2t[T[t, k]].append((t, k))

    def patch_key(self, a):
        # patches with equal keys have equal local matrices
        patch = self.v2t[a]
        rel = (self.V[self.T[[t for t, _ in patch]]] - self.V[a]) / self.h
        return (tuple(np.round(rel.ravel(), 9)), tuple(k for _, k in patch),
                tuple(bool(self.bnd_edge[self.TE[t, e]]) for t, _ in patch for e in range(3)), bool(self.bnd[a]))

    def values(self, u):
        return np.einsum("qk,tk...->tq...", self.L, u[self.T])

    def grad(self, u):
        return np.einsum("tki,tk->ti", self.G, u[self.T])

    def exact(self, u, grad):
        # L2-normalised exact eigenfunction and its gradient at the quadrature points
        U = u(self.X)
        nrm = np.sqrt(np.sum(self.W * U**2))
        return U / nrm, grad(self.X) / nrm


def rt_basis(k, xi):
    # RT_k = [P_k]^2 + x P_k(homogeneous), monomials in the local coordinates xi
    ex = [(c, a, b) for a in range(k + 1) for b in range(k + 1 - a) for c in (0, 1)]
    ex += [(2, a, k - a) for a in range(k + 1)]
    x, y = xi[:, 0], xi[:, 1]
    Phi = np.zeros((len(xi), len(ex), 2))
    Div = np.zeros((len(xi), len(ex)))
    for j, (c, a, b) in enumerate(ex):
        mono = x**a * y**b
        if c == 0:
            Phi[:, j, 0] = mono
            Div[:, j] = a * x**(a - 1) * y**b if a else 0
        elif c == 1:
            Phi[:, j, 1] = mono
            Div[:, j] = b * x**a * y**(b - 1) if b else 0
        else:
            Phi[:, j, 0] = x * mono
            Phi[:, j, 1] = y * mono
            Div[:, j] = (a + b + 2) * mono
    return Phi, Div


def monomials(k, xi):
    return np.stack([xi[:, 0]**a * xi[:, 1]**b for a in range(k + 1) for b in range(k + 1 - a)], axis=1)


def equilibrated_flux(m, uh, lam, k=1):
    """Patchwise equilibrated flux of Cances et al. (2017) in RT_k.

    Returns eta = ||grad u_h + sigma|| and sigma at the quadrature points."""
    nt, nq = m.nt, m.nq
    nb, nm = (k + 1) * (k + 3), (k + 1) * (k + 2) // 2
    xi = ((m.X - m.cK[:, None]) / m.sK[:, None, None]).reshape(-1, 2)
    Phi, Div = rt_basis(k, xi)
    Phi = Phi.reshape(nt, nq, nb, 2)
    Div = Div.reshape(nt, nq, nb) / m.sK[:, None, None]
    Pk = monomials(k, xi).reshape(nt, nq, nm)
    H = np.einsum("tq,tqjd,tqld->tjl", m.W, Phi, Phi)
    D = np.einsum("tq,tqj,tqm->tmj", m.W, Div, Pk)
    Uq, Gu = m.values(uh), m.grad(uh)

    # normal traces at k + 1 Gauss points per edge, oriented by the global edge
    s = (np.polynomial.legendre.leggauss(k + 1)[0] + 1) / 2
    Nrm = np.zeros((nt, 3, k + 1, nb))
    for e in range(3):
        A0, A1 = m.V[m.E[m.TE[:, e], 0]], m.V[m.E[m.TE[:, e], 1]]
        d = A1 - A0
        normal = np.stack([d[:, 1], -d[:, 0]], axis=1) / np.linalg.norm(d, axis=1)[:, None]
        for g in range(k + 1):
            Ph, _ = rt_basis(k, (A0 + s[g] * d - m.cK) / m.sK[:, None])
            Nrm[:, e, g] = np.einsum("tjd,td->tj", Ph, normal)

    sigma = np.zeros((nt, nb))
    cache = {}
    for a in range(m.nv):
        patch = m.v2t[a]
        n = len(patch) * nb
        key = m.patch_key(a)
        if key not in cache:
            # constraints: divergence moments on each element, continuous normal trace inside the patch,
            # zero normal trace on the patch boundary except on the Dirichlet boundary
            Hp = np.zeros((n, n))
            rows = []
            for i, (t, _) in enumerate(patch):
                Hp[i * nb:(i + 1) * nb, i * nb:(i + 1) * nb] = H[t]
                for q in range(nm):
                    row = np.zeros(n)
                    row[i * nb:(i + 1) * nb] = D[t, q]
                    rows.append(row)
            shared = {}
            for i, (t, _) in enumerate(patch):
                for e in range(3):
                    shared.setdefault(m.TE[t, e], []).append((i, t, e))
            for edge, lst in shared.items():
                if m.bnd_edge[edge]:
                    continue
                for g in range(k + 1):
                    row = np.zeros(n)
                    for sgn, (i, t, e) in zip([1, -1], lst):
                        row[i * nb:(i + 1) * nb] = sgn * Nrm[t, e, g]
                    rows.append(row)
            # minimise ||sigma + psi_a grad u_h|| under the constraints (null-space method)
            Uc, Sc, Vt = np.linalg.svd(np.array(rows))
            r = int((Sc > Sc.max() * 1e-10).sum())
            pinv = (Vt[:r].T / Sc[:r]) @ Uc[:, :r].T
            Z = Vt[r:].T
            R = Z @ np.linalg.inv(Z.T @ Hp @ Z) @ Z.T
            cache[key] = ((np.eye(n) - R @ Hp) @ pinv[:, :len(patch) * nm], R)
        Ad, R = cache[key]
        f, b = np.zeros(len(patch) * nm), np.zeros(n)
        for i, (t, j) in enumerate(patch):
            psi = m.L[:, j]
            f[i * nm:(i + 1) * nm] = (m.W[t] * (lam * Uq[t] * psi - Gu[t] @ m.G[t, j])) @ Pk[t]
            b[i * nb:(i + 1) * nb] = np.einsum("q,qjd,d->j", m.W[t] * psi, Phi[t], Gu[t])
        c = Ad @ f - R @ b
        for i, (t, _) in enumerate(patch):
            sigma[t] += c[i * nb:(i + 1) * nb]
    Sq = np.einsum("tj,tqjd->tqd", sigma, Phi)
    eta = np.sqrt(np.sum(m.W * np.sum((Gu[:, None] + Sq)**2, axis=2)))
    return eta, Sq


def p2_basis(L, G):
    N = np.zeros((len(L), 6))
    dN = np.zeros((len(G), len(L), 6, 2))
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        N[:, i] = L[:, i] * (2 * L[:, i] - 1)
        N[:, 3 + i] = 4 * L[:, j] * L[:, k]
        dN[:, :, i] = (4 * L[:, i] - 1)[None, :, None] * G[:, i][:, None]
        dN[:, :, 3 + i] = 4 * (L[:, k][None, :, None] * G[:, j][:, None] + L[:, j][None, :, None] * G[:, k][:, None])
    return N, dN


def residual_lifting(m, uh, lam):
    """Lower bound <Res, r>^2 / ||grad r||^2 <= ||Res||_{-1}^2, r = sum_a psi_a r_a with r_a in P_2 on each patch."""
    T, G, L, W = m.T, m.G, m.L, m.W
    N, dN = p2_basis(L, G)
    A6 = np.einsum("tq,tqid,tqjd->tij", W, dN, dN)
    m6 = W @ N
    Uq, Gu = m.values(uh), m.grad(uh)
    rq, grq = np.zeros((m.nt, m.nq)), np.zeros((m.nt, m.nq, 2))
    cache = {}
    for a in range(m.nv):
        patch = m.v2t[a]
        dofs, ids = {}, []
        for t, _ in patch:
            ids.append([dofs.setdefault(("v", T[t, i]), len(dofs)) for i in range(3)]
                       + [dofs.setdefault(("e", m.TE[t, i]), len(dofs)) for i in range(3)])
        nd = len(dofs)
        key = m.patch_key(a)
        if key not in cache:
            A, mean = np.zeros((nd, nd)), np.zeros(nd)
            for (t, _), d in zip(patch, ids):
                A[np.ix_(d, d)] += A6[t]
                mean[d] += m6[t]
            if m.bnd[a]:
                # zero on the boundary of the domain
                keep = np.array([not (m.bnd[i] if typ == "v" else m.bnd_edge[i]) for typ, i in dofs])
                Ainv = np.zeros((nd, nd))
                Ainv[np.ix_(keep, keep)] = np.linalg.inv(A[np.ix_(keep, keep)])
            else:
                # Neumann problem with zero mean
                Kk = np.zeros((nd + 1, nd + 1))
                Kk[:nd, :nd], Kk[:nd, nd], Kk[nd, :nd] = A, mean, mean
                Ainv = np.linalg.inv(Kk)[:nd, :nd]
            cache[key] = Ainv
        f = np.zeros(nd)
        for (t, k), d in zip(patch, ids):
            psi = L[:, k]
            f[d] += (W[t] * (lam * Uq[t] * psi - Gu[t] @ G[t, k])) @ N - np.einsum("q,qid,d->i", W[t] * psi, dN[t], Gu[t])
        r = cache[key] @ f
        for (t, k), d in zip(patch, ids):
            psi = L[:, k]
            ra = N @ r[d]
            rq[t] += psi * ra
            grq[t] += ra[:, None] * G[t, k] + psi[:, None] * np.einsum("qid,i->qd", dN[t], r[d])
    res_r = np.sum(W * lam * Uq * rq) - np.sum(W * np.einsum("tqd,td->tq", grq, Gu))
    return res_r**2 / np.sum(W * np.sum(grq**2, axis=2))


def cluster_errors(m, Uh, lam_h, Uex, lam):
    """E^2 = ||A^(1/2)(g0 - gh)||^2, eps = ||g0 - gh||^2 (Hilbert-Schmidt) and O[i, j] = (u_ih, u_j)."""
    O = np.einsum("tq,tqi,jtq->ij", m.W, m.values(Uh), np.array(Uex))
    eps = 2 * len(lam) - 2 * np.sum(O**2)
    E2 = np.sum(lam) + np.sum(lam_h) - 2 * np.sum(O**2 * lam)
    return E2, eps, O


def Q_h(lam_m, lam_M, E2, eps):
    return 3 * (lam_M - lam_m)**2 / (4 * lam_m) * eps**2 \
        + 3 / lam_m * (1 + eps**2 / 4) * (2 * (1 + lam_M * eps / (4 * lam_m))**2 * E2**2 + 2 * lam_M**2 * eps**2)


def angle_matrix(m, K, free, Gex, lam):
    """D[k, l] = a((I - P_h) psi_k, (I - P_h) psi_l) for the a-normalised exact eigenfunctions psi_k.

    delta_{h,p} of Eq. (6) is the square root of the largest eigenvalue of D[:p, :p]."""
    lu = spla.splu(K[free][:, free].tocsc())
    X = np.zeros((m.nv, len(Gex)))
    for k, g in enumerate(Gex):
        rhs = np.zeros(m.nv)
        np.add.at(rhs, m.T.ravel(), np.einsum("tq,tqd,tjd->tj", m.W, g, m.G).ravel())
        X[free, k] = lu.solve(rhs[free])
    s = 1 / np.sqrt(lam)
    return np.eye(len(Gex)) - s[:, None] * (X.T @ (K @ X)) * s[None, :]


def angle(D, p):
    return np.sqrt(max(np.linalg.eigvalsh(D[:p, :p]).max(), 0)) if p else 0.0
