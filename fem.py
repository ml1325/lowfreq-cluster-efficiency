import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def mesh_triangle(N):
    # triangle (0,0), (1,0), (0,1) with N subdivisions per leg
    idx = -np.ones((N + 1, N + 1), dtype=int)
    V = []
    for j in range(N + 1):
        for i in range(N + 1 - j):
            idx[i, j] = len(V)
            V.append((i / N, j / N))
    T = []
    for j in range(N):
        for i in range(N - j):
            T.append((idx[i, j], idx[i + 1, j], idx[i, j + 1]))
            if i + j + 2 <= N:
                T.append((idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]))
    V = np.array(V)
    x, y = V.T
    bnd = (x < 1e-12) | (y < 1e-12) | (abs(x + y - 1) < 1e-12)
    return V, np.array(T), bnd, idx


def mesh_rect(a, Nx, Ny):
    # rectangle (0,a) x (0,1)
    idx = np.arange((Nx + 1) * (Ny + 1)).reshape(Nx + 1, Ny + 1)
    X, Y = np.meshgrid(np.linspace(0, a, Nx + 1), np.linspace(0, 1, Ny + 1), indexing="ij")
    V = np.column_stack([X.ravel(), Y.ravel()])
    T = []
    for i in range(Nx):
        for j in range(Ny):
            T.append((idx[i, j], idx[i + 1, j], idx[i, j + 1]))
            T.append((idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]))
    x, y = V.T
    bnd = (x < 1e-12) | (y < 1e-12) | (abs(x - a) < 1e-12) | (abs(y - 1) < 1e-12)
    return V, np.array(T), bnd, idx


def geometry(V, T):
    # areas and gradients of the barycentric coordinates, G[t, k] = grad lambda_k
    P = V[T]
    d1, d2 = P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]
    det = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    G = np.empty((len(T), 3, 2))
    for k in range(3):
        a, b = P[:, (k + 1) % 3], P[:, (k + 2) % 3]
        G[:, k, 0] = (a[:, 1] - b[:, 1]) / det
        G[:, k, 1] = (b[:, 0] - a[:, 0]) / det
    return det / 2, G


def edges(T):
    # E[e] = (v0, v1) with v0 < v1, TE[t, k] = edge opposite vertex k
    loc = np.stack([T[:, [1, 2]], T[:, [2, 0]], T[:, [0, 1]]], axis=1)
    E, TE, count = np.unique(np.sort(loc, axis=2).reshape(-1, 2), axis=0,
                             return_inverse=True, return_counts=True)
    return E, TE.reshape(len(T), 3), count == 1


def sparse(Kloc, dofs, n):
    rows, cols = np.repeat(dofs, 3, axis=1).ravel(), np.tile(dofs, (1, 3)).ravel()
    return sp.csr_matrix((Kloc.ravel(), (rows, cols)), shape=(n, n))


def assemble_p1(V, T):
    area, G = geometry(V, T)
    K = np.einsum("tki,tli->tkl", G, G) * area[:, None, None]
    M = (np.ones((3, 3)) + np.eye(3)) / 12 * area[:, None, None]
    return sparse(K, T, len(V)), sparse(M, T, len(V))


def eig_p1(K, M, free, k, sigma):
    Kf, Mf = K[free][:, free].tocsc(), M[free][:, free].tocsc()
    lam, X = spla.eigsh(Kf, k=k, M=Mf, sigma=sigma, which="LM", tol=1e-12)
    order = np.argsort(lam)
    lam, X = lam[order], X[:, order]
    X /= np.sqrt(np.einsum("ij,ij->j", X, Mf @ X))
    res = np.linalg.norm(Kf @ X - (Mf @ X) * lam, axis=0)
    U = np.zeros((K.shape[0], k))
    U[free] = X
    return lam, U, res


def cr_lower_bounds(V, T, k, kappa=0.1893):
    # Carstensen and Gedicke (2014): lambda_CR / (1 + kappa^2 H^2 lambda_CR) <= lambda
    area, G = geometry(V, T)
    E, TE, bnd_edge = edges(T)
    K = sparse(4 * np.einsum("tki,tli->tkl", G, G) * area[:, None, None], TE, len(E))
    M = sparse(np.eye(3) * (area / 3)[:, None, None], TE, len(E))
    free = ~bnd_edge
    lam = np.sort(spla.eigsh(K[free][:, free].tocsc(), k=k, M=M[free][:, free].tocsc(), sigma=0.0,
                             which="LM", tol=1e-12)[0])
    P = V[T]
    H = max(np.linalg.norm(P[:, i] - P[:, j], axis=1).max() for i, j in [(1, 0), (2, 0), (2, 1)])
    return lam / (1 + kappa**2 * H**2 * lam)


def dunavant(degree):
    # symmetric rules of Dunavant (1985), barycentric points; entries (a, b, weight, orbit size)
    if degree == 4:
        rule = [(0, 0.445948490915965, 0.223381589678011, 3),
                (0, 0.091576213509771, 0.109951743655322, 3)]
    else:
        rule = [(1 / 3, 1 / 3, 0.144315607677787, 1),
                (0, 0.459292588292723, 0.095091634267285, 3),
                (0, 0.170569307751760, 0.103217370534718, 3),
                (0, 0.050547228317031, 0.032458497623198, 3),
                (0.008394777409958, 0.263112829634638, 0.027230314174435, 6)]
    pts, w = [], []
    for a, b, wt, n in rule:
        if n == 1:
            orbit = [(a, a, a)]
        elif n == 3:
            c = 1 - 2 * b
            orbit = [(c, b, b), (b, c, b), (b, b, c)]
        else:
            c = 1 - a - b
            orbit = [(a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)]
        pts += orbit
        w += [wt] * n
    return np.array(pts), np.array(w)


def prolong(idx_c, idx_f):
    # P1 interpolation from a structured grid to its uniform refinement
    rows, cols, vals = [], [], []
    for i, j in zip(*np.nonzero(idx_f >= 0)):
        if i % 2 == 0 and j % 2 == 0:
            parents = [(i // 2, j // 2)]
        elif j % 2 == 0:
            parents = [((i - 1) // 2, j // 2), ((i + 1) // 2, j // 2)]
        elif i % 2 == 0:
            parents = [(i // 2, (j - 1) // 2), (i // 2, (j + 1) // 2)]
        else:
            parents = [((i + 1) // 2, (j - 1) // 2), ((i - 1) // 2, (j + 1) // 2)]
        for p in parents:
            rows.append(idx_f[i, j])
            cols.append(idx_c[p])
            vals.append(1 / len(parents))
    return sp.csr_matrix((vals, (rows, cols)), shape=(idx_f.max() + 1, idx_c.max() + 1))


def triangle_modes(n):
    # lambda = pi^2 (j^2 + l^2), j > l >= 1
    modes = sorted((np.pi**2 * (j * j + l * l), j, l) for j in range(2, 9) for l in range(1, j))
    out = []
    for lam, j, l in modes[:n]:
        def u(X, j=j, l=l):
            x, y = X[..., 0], X[..., 1]
            return np.sin(j * np.pi * (1 - y)) * np.sin(l * np.pi * x) - np.sin(l * np.pi * (1 - y)) * np.sin(j * np.pi * x)

        def grad(X, j=j, l=l):
            x, y = X[..., 0], X[..., 1]
            ux = l * np.pi * np.sin(j * np.pi * (1 - y)) * np.cos(l * np.pi * x) \
                - j * np.pi * np.sin(l * np.pi * (1 - y)) * np.cos(j * np.pi * x)
            uy = -j * np.pi * np.cos(j * np.pi * (1 - y)) * np.sin(l * np.pi * x) \
                + l * np.pi * np.cos(l * np.pi * (1 - y)) * np.sin(j * np.pi * x)
            return np.stack([ux, uy], axis=-1)
        out.append((lam, u, grad))
    return out


def rect_modes(a, n):
    # lambda = pi^2 (j^2 / a^2 + l^2)
    modes = sorted((np.pi**2 * (j * j / a**2 + l * l), j, l) for j in range(1, 9) for l in range(1, 9))
    out = []
    for lam, j, l in modes[:n]:
        def u(X, j=j, l=l):
            return np.sin(j * np.pi * X[..., 0] / a) * np.sin(l * np.pi * X[..., 1])

        def grad(X, j=j, l=l):
            x, y = X[..., 0], X[..., 1]
            return np.stack([j * np.pi / a * np.cos(j * np.pi * x / a) * np.sin(l * np.pi * y),
                             l * np.pi * np.sin(j * np.pi * x / a) * np.cos(l * np.pi * y)], axis=-1)
        out.append((lam, u, grad))
    return out
