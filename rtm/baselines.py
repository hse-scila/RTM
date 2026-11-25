"""Neural baselines: ProdLDA (Srivastava, Sutton 2017, AVITM) and ETM (Dieng, Ruiz, Blei 2020), configured as in
the original papers and not tuned per corpus. Both are trained 100 epochs with batches of 256; ETM with a linear KL
warm-up over the first 20 epochs. Topics are taken after the last epoch.
ProdLDA topic = word distribution of a one-topic document in evaluation mode, softmax(BN(beta_k)).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .data import load_bow


class ProdLDA(nn.Module):
    def __init__(self, V, T, hidden=100, drop=0.2):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(V, hidden), nn.Linear(hidden, hidden)
        self.mu, self.lv = nn.Linear(hidden, T), nn.Linear(hidden, T)
        self.bn_mu, self.bn_lv = nn.BatchNorm1d(T, affine=False), nn.BatchNorm1d(T, affine=False)
        self.drop, self.drop_theta = nn.Dropout(drop), nn.Dropout(drop)
        self.beta = nn.Parameter(torch.empty(T, V))
        nn.init.xavier_uniform_(self.beta)
        self.bn_dec = nn.BatchNorm1d(V, affine=False)
        a = torch.ones(T)                                  # Dirichlet(1), Laplace approximation
        self.register_buffer("pm", torch.log(a) - torch.log(a).mean())
        self.register_buffer("pv", (1 / a) * (1 - 2 / T) + (1 / a).sum() / T ** 2)

    def forward(self, x):
        h = self.drop(Fn.softplus(self.fc2(Fn.softplus(self.fc1(x)))))
        mu, lv = self.bn_mu(self.mu(h)), self.bn_lv(self.lv(h))
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv) if self.training else mu
        theta = self.drop_theta(torch.softmax(z, 1))
        logp = torch.log_softmax(self.bn_dec(theta @ self.beta), 1)
        nll = -(x * logp).sum(1)
        var = torch.exp(lv)
        kl = 0.5 * ((var / self.pv).sum(1) + ((mu - self.pm) ** 2 / self.pv).sum(1) - mu.shape[1]
                    + torch.log(self.pv).sum() - lv.sum(1))
        return nll, kl

    def phi(self):
        """Word distribution of a one-topic document (θ = e_k) in evaluation mode: softmax(BN(β_k)) with the running
        BatchNorm statistics. softmax(β) itself is almost uniform because the decoder BatchNorm removes the word scale."""
        return torch.softmax(self.bn_dec(self.beta), 1)


class ETM(nn.Module):
    def __init__(self, V, T, hidden=800, emb=300):
        super().__init__()
        self.q = nn.Sequential(nn.Linear(V, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.mu, self.ls = nn.Linear(hidden, T), nn.Linear(hidden, T)
        self.rho = nn.Linear(emb, V, bias=False)        # word embeddings V × emb
        self.alpha = nn.Linear(emb, T, bias=False)      # topic embeddings T × emb

    def phi(self):
        return torch.softmax(self.alpha(self.rho.weight).T, 1)    # T × V

    def forward(self, x):
        xn = x / x.sum(1, keepdim=True).clamp_min(1)
        h = self.q(xn)
        mu, ls = self.mu(h), self.ls(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * ls) if self.training else mu
        theta = torch.softmax(z, 1)
        p = (theta @ self.phi()).clamp_min(1e-10)
        nll = -(x * torch.log(p)).sum(1)
        kl = -0.5 * (1 + ls - mu ** 2 - ls.exp()).sum(1)
        return nll, kl


CFG = {"prodlda": dict(cls=ProdLDA, lr=2e-3, betas=(0.99, 0.999), wd=0.0),
       "etm": dict(cls=ETM, lr=5e-3, betas=(0.9, 0.999), wd=1.2e-6)}



def train_vae(model_name, data_dir, T, seed, out_dir, epochs=100, batch=256, kl_warmup=None, device=None):
    """model_name: "prodlda" or "etm". kl_warmup defaults to 20 epochs for ETM and 0 for ProdLDA (as in the paper).
    Saves phi.npy, model.pt, train_log.csv."""
    if kl_warmup is None:
        kl_warmup = 20 if model_name == "etm" else 0
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X, _ = load_bow(data_dir)
    D, V = X.shape
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    c = CFG[model_name]
    model = c["cls"](V, T).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=c["lr"], betas=c["betas"], weight_decay=c["wd"])
    log, t0 = [], time.time()
    for ep in range(epochs):
        model.train()
        tot, tok, kls = 0.0, 0.0, 0.0
        w = min(1.0, (ep + 1) / kl_warmup) if kl_warmup else 1.0
        idx = rng.permutation(D)
        for s in range(0, D, batch):
            rows = idx[s:s + batch]
            if len(rows) < 2:                              # BatchNorm needs at least two documents
                continue
            x = torch.from_numpy(X[rows].toarray()).to(dev)
            nll, kl = model(x)
            loss = (nll + w * kl).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot, tok, kls = tot + nll.sum().item(), tok + x.sum().item(), kls + kl.sum().item()
        log.append({"epoch": ep + 1, "loss": (tot + kls) / D, "ppl": float(np.exp(tot / tok)), "kl": kls / D,
                    "sec": round(time.time() - t0, 1)})
    model.eval()
    with torch.no_grad():
        np.save(out / "phi.npy", model.phi().double().cpu().numpy())
    torch.save(model.state_dict(), out / "model.pt")
    pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)
    return model, pd.DataFrame(log)
