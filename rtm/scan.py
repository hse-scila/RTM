"""Grid scans: layout of trained models and computation of the criterion curves.

Layout: models/<model>/<corpus>/s<seed>/T<T>/ with phi.npy (RTM, ProdLDA, ETM) or phi200.csv (LDA);
<model> is one of rtm, lda, prodlda, etm.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import criteria as cr
from .data import CORPORA, DATA, ROOT, background, load_bow, t_max, word_vectors
from .lda import load_phi

MODELS_DIR = ROOT / "models"
NAMES = {"RTM": "rtm", "LDA": "lda", "ProdLDA": "prodlda", "ETM": "etm"}


def model_dir(model, corpus, seed, T, root=MODELS_DIR):
    return Path(root) / NAMES[model] / corpus / f"s{seed}" / f"T{T}"


def load_topics(model, d, vocab):
    """Topics of one trained model or None. RTM runs whose final loss exceeds 100 are not accepted (paper, 6.10)."""
    d = Path(d)
    if model == "RTM":
        if (d / "phi.npy").exists() and pd.read_csv(d / "train_log.csv")["loss"].iloc[-1] < 100:
            return np.load(d / "phi.npy")
        return None
    if model == "LDA":
        return load_phi(d / "phi200.csv", vocab) if (d / "phi200.csv").exists() else None
    return np.load(d / "phi.npy") if (d / "phi.npy").exists() else None


def compute_results(root=MODELS_DIR, corpora=None, models=("RTM", "LDA", "ProdLDA", "ETM"), out_dir=None):
    """Energy and silhouette of every model, median JS of every T with three valid runs, recovery of true topics
    on synthetic corpora. Writes curves.csv, stability.csv, recovery.csv to out_dir (default results/)."""
    cur, sta, rec = [], [], []
    for c in corpora or CORPORA:
        X, vocab = load_bow(DATA / c)
        bg = background(X)
        E = word_vectors(c)
        P = np.load(DATA / c / "phi_true.npy") if c.startswith("synth") else None
        for m in models:
            base = Path(root) / NAMES[m] / c
            Ts = sorted({int(d.name[1:]) for d in base.glob("s*/T*") if d.name[1:].isdigit()})
            for T in Ts:
                phis = {}
                for s in range(3):
                    phi = load_topics(m, model_dir(m, c, s, T, root), vocab)
                    if phi is None:
                        continue
                    phis[s] = phi
                    cur.append({"corpus": c, "model": m, "seed": s, "T": T, "E": cr.energy(phi, m, bg),
                                "silhouette": cr.silhouette(phi, bg, E)})
                    if P is not None and m in ("RTM", "LDA"):
                        rec.append({"corpus": c, "model": m, "seed": s, "T": T, "found": cr.recovery(phi, P)})
                if len(phis) == 3 and 4 <= T <= t_max(c):
                    for rep in ("full", "lift"):
                        sta.append({"corpus": c, "model": m, "representation": rep, "T": T,
                                    "median_JS": cr.median_js(phis, rep, bg)})
            print(c, m, len(Ts), "T values", flush=True)
    out = Path(out_dir or ROOT / "results")
    out.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(cur), pd.DataFrame(sta), pd.DataFrame(rec)
    for x, f in zip(res, ("curves", "stability", "recovery")):
        x.to_csv(out / f"{f}.csv", index=False)
    return res
