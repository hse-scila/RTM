"""Prepared corpora: data/<corpus>/{bow.npz, vocab.txt, stats.json, docs.txt.gz, w2v_vectors.npy, ...}."""
import gzip
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# corpus -> (title, K = number of generating topics or rubrics, language)
CORPORA = {
    "synth_K6": ("Synthetic, 6 equal topics", 6, "—"),
    "synth_k": ("Synthetic, 12 equal topics", 12, "—"),
    "synth_K20": ("Synthetic, 20 equal topics", 20, "—"),
    "synth_K12_uneq": ("Synthetic, 12 unequal topics", 12, "—"),
    "lenta_E2": ("Lenta", 10, "ru"),
    "news20_clean": ("20 Newsgroups", 20, "en"),
    "livedoor": ("Livedoor", 9, "ja"),
    "thucnews": ("THUCNews", 14, "zh"),
    "gnad10": ("GNAD10", 9, "de"),
    "sanad": ("SANAD", 7, "ar"),
    "mnds": ("MN-DS", 17, "en"),
}
SYNTH = [c for c in CORPORA if c.startswith("synth")]
REAL = [c for c in CORPORA if not c.startswith("synth")]
SEEDS = (0, 1, 2)
T_MIN = 4
T_MAX = {"synth_K20": 40}            # upper end of the analysed grid; 30 for all other corpora


def t_max(corpus):
    return T_MAX.get(corpus, 30)


def grid(corpus, model="RTM"):
    """Grid of T used in the paper: 4…30 with step 2 plus step 1 in the window K − 2 … K + 2.
    Synthetic corpora have the extra points listed in Table 2."""
    K = CORPORA[corpus][1]
    base = set(range(4, 31, 2)) | set(range(K - 2, K + 3))
    if corpus == "synth_K6" and model == "RTM":
        base = set(range(4, 11)) | {12, 14, 16, 18, 20, 24, 30}
    if corpus == "synth_K20":
        base = ({4, 8} | set(range(12, 29, 2)) | {32, 36, 40} | {19, 21}) if model == "RTM" else \
               (set(range(4, 41, 2)) | {19, 21})
    if corpus.startswith("synth") and model == "LDA" and corpus != "synth_k":
        base |= set(range(32, 41, 2))
    return sorted(t for t in base if t >= T_MIN)


def epochs_rtm(corpus):
    return 400 if corpus.startswith("synth") else 450


def load_bow(data_dir):
    data_dir = Path(data_dir)
    X = sp.load_npz(data_dir / "bow.npz").tocsr()
    vocab = (data_dir / "vocab.txt").read_text(encoding="utf-8").split("\n")
    return X, vocab


def save_bow(out_dir, X, vocab):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sp.save_npz(out_dir / "bow.npz", X)
    (out_dir / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")


def background(X):
    """Corpus word frequencies with +1 smoothing, p_bg [V] (float64)."""
    freq = np.asarray(X.sum(0)).ravel() + 1.0
    return freq / freq.sum()


def stats(corpus):
    return json.loads((DATA / corpus / "stats.json").read_text(encoding="utf-8"))


def docs_txt(corpus, out_path=None):
    """Unpack docs.txt (one document per line, space-separated tokens) used as input of Gibbs LDA."""
    out_path = Path(out_path or DATA / corpus / "docs.txt")
    if not out_path.exists():
        with gzip.open(DATA / corpus / "docs.txt.gz", "rb") as f:
            out_path.write_bytes(f.read())
    return out_path


def word_vectors(corpus):
    """word2vec vectors of the corpus vocabulary used for the silhouette (paper, section 4.5), [V, 100] float32,
    rows in vocabulary order, already L2-normalised."""
    return np.load(DATA / corpus / "w2v_vectors.npy")
