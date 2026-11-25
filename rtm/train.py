"""Training of RTM exactly as in the paper (Algorithm 2): Adam with two parameter groups, batches of 256,
nested dropout k_d ~ U{2, ..., k_max}, restart of unused topics every 5 epochs, final model after the last epoch.
With the same seed and device the run is reproduced bit for bit.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import background, load_bow
from .model import ResidualTopicModel


def _batches(X, idx, size, dev):
    for s in range(0, len(idx), size):
        rows = idx[s:s + size]
        yield rows, torch.from_numpy(X[rows].toarray()).to(dev)


def train_rtm(data_dir, T, seed, out_dir, epochs=450, k_max=10, k_min=2, d=256, hidden=512, batch=256,
              lr=1e-3, lr_topics=3e-2, commit_weight=0.25, restart_every=5, threads=2, device=None, verbose=True):
    """Train one RTM and save model.pt, train_log.csv, phi.npy (full topics) and phi_own.npy (own parts).
    epochs: 400 on synthetic corpora, 450 on real corpora in the paper."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    k_max = min(k_max, T)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X, _ = load_bow(data_dir)
    D, V = X.shape
    model = ResidualTopicModel(V, T, d, hidden, torch.from_numpy(np.log(background(X)))).to(dev)
    rest = [p for n, p in model.named_parameters() if n != "beta"]
    opt = torch.optim.Adam([{"params": rest}, {"params": [model.beta], "lr": lr_topics}], lr=lr)
    usage = torch.zeros(T, device=dev)
    log, t0 = [], time.time()
    for ep in range(epochs):
        tot, tot_nll, n_b, depths = 0.0, 0.0, 0, []
        for rows, x in _batches(X, rng.permutation(D), batch, dev):
            u = model.encode(x)
            chosen, alpha = model.pursuit(u, k_max)
            k = torch.randint(k_min, k_max + 1, (x.shape[0],), device=dev)
            loss, nll = model.loss(x, u, chosen, alpha, k, commit_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            act = torch.arange(k_max, device=dev)[None] < k[:, None]
            usage.index_add_(0, chosen[act], torch.ones(int(act.sum()), device=dev))
            tot, tot_nll, n_b = tot + loss.item(), tot_nll + nll.item(), n_b + 1
            depths.append(k.cpu())
        dead = 0
        if restart_every and (ep + 1) % restart_every == 0 and ep + 1 < epochs:      # Algorithm 3
            dead_idx = torch.nonzero(usage == 0).flatten()
            dead = len(dead_idx)
            if dead:
                with torch.no_grad():
                    rows = rng.choice(D, size=dead, replace=False)
                    xd = torch.from_numpy(X[rows].toarray()).to(dev)
                    u = model.encode(xd)
                    model.codes[dead_idx] = u + 0.01 * torch.randn_like(u)
                    p_doc = (xd + 0.01) / (xd + 0.01).sum(1, keepdim=True)
                    model.beta[dead_idx] = 0.5 * (torch.log(p_doc) - model.b)
            usage.zero_()
        rec = {"epoch": ep + 1, "loss": tot / n_b, "ppl": float(np.exp(tot_nll / n_b)),
               "k_mean": float(torch.cat(depths).float().mean()), "restarted": dead, "sec": round(time.time() - t0, 1)}
        log.append(rec)
        if verbose and (ep % 50 == 0 or ep == epochs - 1):
            print(json.dumps(rec), flush=True)
    torch.save({"state": model.state_dict(), "args": dict(T=T, d=d, hidden=hidden, k_max=k_max, seed=seed)},
               out / "model.pt")
    pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)
    with torch.no_grad():
        np.save(out / "phi.npy", model.phi().double().cpu().numpy())
        np.save(out / "phi_own.npy", model.phi_own().double().cpu().numpy())
    return model, pd.DataFrame(log)


def load_rtm(run_dir, V, device="cpu"):
    ck = torch.load(Path(run_dir) / "model.pt", map_location=device, weights_only=False)
    a = ck["args"]
    m = ResidualTopicModel(V, a["T"], a["d"], a["hidden"]).to(device)
    m.load_state_dict(ck["state"])
    m.eval()
    return m
