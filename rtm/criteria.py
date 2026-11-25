"""Three criteria of the number of topics (paper, section 4): energy knee, minimum of the median Jensen–Shannon
divergence between runs, maximum of the top-word silhouette; plus topic recovery on synthetic data."""
from itertools import combinations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import rel_entr


# ---------------------------------------------------------------- density of states and energy (sections 4.1–4.2)
def renyi_parts(phi, thr, rel=False):
    """thr — level of a "high" cell: a number (1/V for LDA, ProdLDA, ETM) or a vector p_bg [V] (RTM).
    rel=True — energy relative to the mass of the same cells in the initial state (RTM, eq. 13).
    Returns (Renyi entropy S_R, g = −ln ρ, energy E)."""
    T, V = phi.shape
    high = phi > thr
    E = -np.log(phi[high].sum() / T)
    if rel:
        E = -np.log(phi[high].sum() / (np.broadcast_to(thr, phi.shape)[high]).sum())
    g = -np.log(high.sum() / (V * T))
    return (E + T * g) / (T - 1), g, E


def energy(phi, model, p_bg):
    """Energy as in the paper: RTM from its initial state (the background), other models from 1/V."""
    if model == "RTM":
        return renyi_parts(phi, p_bg, rel=True)[2]
    return renyi_parts(phi, 1 / phi.shape[1])[2]


def knee(t, y):
    """Knee in bilogarithmic coordinates (eqs. 14–15): two least-squares segments over ln T, the joint belongs to
    both, at least three points per segment. Returns (T*, slope before, slope after)."""
    t = np.asarray(t, float)
    x = np.log(t)
    y = np.asarray(y, float)
    best = None
    for i in range(2, len(t) - 2):
        sse, sl = 0.0, []
        for a, b in ((0, i + 1), (i, len(t))):
            k, c = np.polyfit(x[a:b], y[a:b], 1)
            sse += ((np.polyval([k, c], x[a:b]) - y[a:b]) ** 2).sum()
            sl.append(k)
        if best is None or sse < best[0]:
            best = (sse, t[i], sl)
    return best[1], best[2][0], best[2][1]


# ---------------------------------------------------------------- reproducibility between runs (section 4.5)
def normalize(p):
    p = np.asarray(p, dtype=np.float64)
    return p / p.sum(1, keepdims=True)


def lift(phi, p_bg):
    """Normalised lift q_t(w) ∝ φ_t(w) / p_bg(w)."""
    return normalize(phi / p_bg)


def js_matrix(P, Q):
    """Jensen–Shannon divergence in bits between all rows of P and Q."""
    d = np.empty((len(P), len(Q)))
    for i, a in enumerate(P):
        mid = (a[None, :] + Q) / 2
        d[i] = (rel_entr(a[None, :], mid).sum(1) + rel_entr(Q, mid).sum(1)) / (2 * np.log(2))
    return np.maximum(d, 0)


def matched_js(phis):
    """phis: {run: φ}. JS of topics matched one to one by the Hungarian algorithm for every pair of runs."""
    out = []
    for a, b in combinations(sorted(phis), 2):
        D = js_matrix(phis[a], phis[b])
        i, j = linear_sum_assignment(D)
        out.append(D[i, j])
    return np.concatenate(out)


def median_js(phis, representation="lift", p_bg=None):
    """Median JS over matched topics of all three pairs of runs; representation "full" (φ) or "lift" (q)."""
    pp = {s: normalize(p) if representation == "full" else lift(p, p_bg) for s, p in phis.items()}
    return float(np.median(matched_js(pp)))


# ---------------------------------------------------------------- semantic separation (silhouette, section 4.5)
N_TOP = 10


def silhouette(phi, p_bg, E, n_top=N_TOP):
    """Silhouette of the top words (eqs. 28–33). E — L2-normalised word vectors [V, dim] in vocabulary order;
    top words by relevance 0.6 ln φ + 0.4 ln(φ / p_bg)."""
    T = phi.shape[0]
    rel = 0.6 * np.log(phi + 1e-30) + 0.4 * (np.log(phi + 1e-30) - np.log(p_bg))
    top = np.argsort(-rel, 1)[:, :n_top]
    W = E[top]
    S = np.einsum("tnd,umd->tunm", W, W)
    D = 1 - S
    a = (D[np.arange(T), np.arange(T)].sum(-1)) / (n_top - 1)
    b = D.mean(-1)
    b[np.arange(T), np.arange(T)] = np.inf
    b = b.min(1)
    return float(np.mean((b - a) / np.maximum(a, b)))


def train_word_vectors(X, vocab, workers=8):
    """word2vec skip-gram as in the paper (dim 100, window 20, 10 epochs, all words), on shuffled bags of words.
    Multi-threaded training is not bit-reproducible; the vectors used in the paper are in data/<corpus>/w2v_vectors.npy."""
    from gensim.models import Word2Vec
    rng = np.random.default_rng(0)
    X = X.tocsr()
    sents = []
    for i in range(X.shape[0]):
        row = X[i]
        toks = np.repeat(row.indices, row.data.astype(int))
        rng.shuffle(toks)
        sents.append([vocab[j] for j in toks])
    m = Word2Vec(sents, vector_size=100, window=20, min_count=1, sg=1, epochs=10, workers=workers, seed=0)
    rng = np.random.default_rng(1)
    E = np.stack([m.wv[w] if w in m.wv else rng.normal(size=m.vector_size) for w in vocab])
    return E / np.linalg.norm(E, axis=1, keepdims=True)


# ---------------------------------------------------------------- recovery of true topics (section 6.1)
def recovery(phi, P_true, thr=0.1):
    """Number of true topics with a matched model topic closer than thr bits of JS (one-to-one matching)."""
    D = js_matrix(normalize(P_true), normalize(phi))
    r, c = linear_sum_assignment(D)
    d = np.full(len(P_true), 1.0)
    d[r] = D[r, c]
    return int(np.sum(d < thr))
