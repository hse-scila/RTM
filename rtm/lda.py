"""LDA with collapsed Gibbs sampling (paper: 200 iterations, alpha = beta = 0.01) through the Gibbs sampler of the
Laboratory for Social and Cognitive Informatics (third_party/pyGibbsLDA2, gibbsldacore.dll).

The sampler is a Windows DLL with a win32 message loop, so this module works on Windows only (pywin32 required).
Each model runs in a separate process with the DLL folder as working directory and PYTHONUTF8=1 (otherwise the
CSV with Arabic, Chinese or Japanese words cannot be written). Output: <out_dir>/phi200.csv, a ';'-separated
table "word; p(w|t_1); …; p(w|t_T)".
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GLDA_DIR = ROOT / "third_party" / "pyGibbsLDA2"
JOB = Path(__file__).resolve().parent / "_glda_job.py"


def train_lda(docs_txt, T, out_dir, iters=200, alpha=0.01, beta=0.01):
    """docs_txt — one document per line, space-separated tokens (rtm.data.docs_txt unpacks it).
    Note: the sampler has its own random initialisation, so repeated runs differ."""
    if sys.platform != "win32":
        raise RuntimeError("gibbsldacore.dll is a Windows library; see README for running LDA on other systems")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    subprocess.run([sys.executable, str(JOB), str(Path(docs_txt).resolve()), str(T), str(iters),
                    str(out_dir.resolve()), str(alpha), str(beta)], cwd=GLDA_DIR, env=env, check=True)
    return out_dir / f"phi{iters}.csv"


def load_phi(path, vocab):
    """phi200.csv -> [T, V] in the order of vocab, rows normalised."""
    ph = pd.read_csv(path, sep=";", header=None, encoding="utf-8", keep_default_na=False)
    idx = {w: i for i, w in enumerate(vocab)}
    words = ph.iloc[:, 0].tolist()
    if len(set(words)) != len(words):
        raise ValueError("duplicate words in the LDA matrix")
    phi = np.zeros((ph.shape[1] - 1, len(vocab)))
    phi[:, [idx[w] for w in words]] = ph.iloc[:, 1:].to_numpy(float).T
    return phi / phi.sum(1, keepdims=True)
