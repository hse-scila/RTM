"""From criterion curves to the estimates and tables of the paper (Tables 11–20).

Inputs (results/*.csv, produced by notebook 04 or shipped with the repository):
  curves.csv     — corpus, model, seed, T, E (energy), silhouette
  stability.csv  — corpus, model, representation (full | lift), T, median_JS  (only T with three valid runs)
Rules (paper, sections 4.5 and 6): T ∈ [4, 30] (up to 40 on synthetic corpus with 20 topics);
energy knee and silhouette — on the grid of each model; minimum of median JS — on the common points where RTM and
LDA both have three valid runs; on ties the smallest T is taken.
"""
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .criteria import knee
from .data import CORPORA, REAL, SYNTH, T_MIN, t_max

MODELS = ["RTM", "LDA", "ProdLDA", "ETM"]
CRIT = ["energy", "js_lift", "js_full", "silhouette"]


def _window(df):
    return df[(df["T"] >= T_MIN) & (df["T"] <= df["corpus"].map(t_max))]


def estimates(curves, stability):
    """One row per (corpus, model): T* of every criterion, values of the JS minima and per-run estimates."""
    curves, stability = _window(curves), _window(stability)
    rows = []
    for c in CORPORA:
        st = stability[stability.corpus == c]
        common = set(st[st.model == "RTM"]["T"]) & set(st[st.model == "LDA"]["T"])
        for m in MODELS:
            x = curves[(curves.corpus == c) & (curves.model == m)]
            if x.empty:
                continue
            mean = x.groupby("T")["E"].mean()
            r = {"corpus": c, "K": CORPORA[c][1], "model": m,
                 "energy": int(knee(mean.index, mean.values)[0]),
                 "energy_runs": "/".join(str(int(knee(g.groupby("T")["E"].mean().index, g.groupby("T")["E"].mean().values)[0]))
                                         for _, g in x.groupby("seed"))}
            for rep in ("lift", "full"):
                q = st[(st.model == m) & (st.representation == rep) & st["T"].isin(common)].set_index("T")["median_JS"]
                q = q.sort_index()
                r[f"js_{rep}"], r[f"js_{rep}_bits"] = int(q.idxmin()), float(q.min())
            s = x.groupby("T")["silhouette"].mean()
            r["silhouette"] = int(s.idxmax())
            r["silhouette_runs"] = "/".join(str(int(g.set_index("T")["silhouette"].idxmax())) for _, g in x.groupby("seed"))
            rows.append(r)
    t = pd.DataFrame(rows)
    for k in CRIT:
        t[f"err_{k}"] = (t[k] - t["K"]).abs()
    return t


def mean_errors(est):
    """Mean |T* − K| over all 11, 4 synthetic and 7 real corpora (Table 19)."""
    out = {}
    for name, cs in (("all 11", list(CORPORA)), ("synthetic 4", SYNTH), ("real 7", REAL)):
        z = est[est.corpus.isin(cs)]
        out[name] = z.groupby("model")[[f"err_{k}" for k in CRIT]].mean().reindex(MODELS)
    return pd.concat(out, names=["corpora"]).round(2)


def hits(est, crit=("energy", "js_lift", "silhouette")):
    """Exact hits and errors ≤ 2 over 33 estimates (11 corpora × 3 criteria), Tables 13 and 19."""
    e = est.melt(id_vars=["model"], value_vars=[f"err_{k}" for k in crit])
    return e.groupby("model")["value"].agg(exact=lambda v: int((v == 0).sum()), within_2=lambda v: int((v <= 2).sum()),
                                           n="count").reindex(MODELS)


def sign_tests(est, against=("LDA", "ProdLDA", "ETM")):
    """Two-sided binomial sign test over corpora, ties excluded (Table 20)."""
    rows = []
    rtm = est[est.model == "RTM"].set_index("corpus")
    for m in against:
        o = est[est.model == m].set_index("corpus")
        for k in CRIT:
            d = (o[f"err_{k}"] - rtm[f"err_{k}"]).dropna()
            w, l = int((d > 0).sum()), int((d < 0).sum())
            p = binomtest(w, w + l).pvalue if w + l else np.nan
            rows.append({"against": m, "criterion": k, "RTM better": w, "worse": l, "ties": int((d == 0).sum()),
                         "p": round(p, 3)})
    return pd.DataFrame(rows)
