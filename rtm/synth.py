"""Synthetic corpora with a known number of topics K (paper, section 5).

True topics are LDA topics (Gibbs sampling, alpha = beta = 0.01, 200 iterations) trained on Lenta with T = K;
they are stored in data/<corpus>/phi_true.npy. A document has k ~ U{1..min(4, K)} topics with shares
theta ~ Dirichlet(2) (minimum share 0.1, sorted in decreasing order); topics are drawn without replacement with
popularity 1/rank (unequal corpus) or uniformly; document lengths are drawn from Lenta; 6000 documents.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp


def make_corpus(phi_true, lengths, D=6000, uneven=False, seed=0, legacy_synth_k=False):
    """phi_true [K, V]; lengths — pool of document lengths (token counts of Lenta documents).
    Returns (X [D, V] CSR, truth DataFrame). legacy_synth_k=True reproduces the generator of the corpus synth_k
    (k ~ U{1..4} drawn as integers(1, 5), topics drawn by rng.choice(K, k) without popularity)."""
    rng = np.random.default_rng(seed)
    K, V = phi_true.shape
    pop = 1.0 / np.arange(1, K + 1) if uneven else np.ones(K)
    pop /= pop.sum()
    rows, cols, vals, truth = [], [], [], []
    for d in range(D):
        k = int(rng.integers(1, 5)) if legacy_synth_k else int(rng.integers(1, min(4, K) + 1))
        while True:
            th = rng.dirichlet(np.full(k, 2.0))
            if th.min() >= 0.1:
                break
        th = np.sort(th)[::-1]
        tops = rng.choice(K, k, replace=False) if legacy_synth_k else rng.choice(K, k, replace=False, p=pop)
        c = rng.multinomial(int(rng.choice(lengths)), th @ phi_true[tops])
        nz = np.flatnonzero(c)
        rows += [d] * len(nz)
        cols += nz.tolist()
        vals += c[nz].tolist()
        truth.append({"k_true": k, "topics": " ".join(map(str, tops)), "theta": " ".join(f"{v:.4f}" for v in th)})
    X = sp.csr_matrix((np.array(vals, np.float32), (rows, cols)), shape=(D, V))
    return X, pd.DataFrame(truth)
