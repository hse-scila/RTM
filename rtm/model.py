"""Residual Topic Model (RTM).

A document is encoded by an MLP into a latent vector u (log(1 + counts) input, centred by a running mean).
Topics are chosen greedily by matching pursuit over a shared codebook of unit vectors e_t:
    t_j = argmax <r, e_t> over topics not chosen yet, alpha_j = max(<r, e_t_j>, 1e-4), r <- r - alpha_j e_t_j.
Each topic has its own word logits beta_t on top of the fixed corpus background b = ln p_bg:
    phi_t = softmax(beta_t + b);  softmax(beta_t) is the "own part" of the topic (= normalised lift phi_t / p_bg).
A document with depth k_d is the mixture p(w|d) = sum_{j <= k_d} theta_j phi_{t_j}(w), theta = alpha / sum(alpha).
Loss = token-normalised NLL + 0.25 * commitment ||u - sum_j alpha_j e_t_j||^2 (mean over documents).
"""
import torch
import torch.nn as nn
import torch.nn.functional as Fn


class ResidualTopicModel(nn.Module):
    def __init__(self, V, T, d=256, hidden=512, background=None, mean_momentum=0.01):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(V, hidden), nn.Softplus(),
            nn.Linear(hidden, hidden), nn.Softplus(),
            nn.Linear(hidden, d),
        )
        self.codes = nn.Parameter(torch.randn(T, d))
        self.beta = nn.Parameter(torch.randn(T, V) * 0.01)
        self.register_buffer("b", torch.zeros(V) if background is None else background.float())
        self.register_buffer("u_mean", torch.zeros(d))
        self.mean_momentum = mean_momentum

    def unit_codes(self):
        return Fn.normalize(self.codes, dim=-1)

    def phi(self):
        """Full topics softmax(beta + b), [T, V]."""
        return torch.softmax(self.beta + self.b, dim=-1)

    def phi_own(self):
        """Own part of the topics softmax(beta), [T, V]."""
        return torch.softmax(self.beta, dim=-1)

    def encode(self, counts):
        u = self.encoder(torch.log1p(counts))
        if self.training:
            with torch.no_grad():
                self.u_mean.lerp_(u.mean(0), self.mean_momentum)
        return u - self.u_mean

    def pursuit(self, u, k_max):
        """Greedy matching pursuit of k_max distinct topics -> (chosen [B, k_max], alpha [B, k_max])."""
        E = self.unit_codes()
        r = u
        used = torch.zeros(u.shape[0], E.shape[0], dtype=torch.bool, device=u.device)
        chosen, alphas = [], []
        for _ in range(k_max):
            s = (r @ E.T).masked_fill(used, float("-inf"))
            t = s.argmax(1)
            a = s.gather(1, t[:, None]).squeeze(1).clamp_min(1e-4)
            r = r - a[:, None] * E[t]
            used = used | Fn.one_hot(t, E.shape[0]).bool()
            chosen.append(t)
            alphas.append(a)
        return torch.stack(chosen, 1), torch.stack(alphas, 1)

    def loss(self, counts, u, chosen, alpha, depth, commit_weight=0.25):
        """depth [B] — number of topics used for each document. Returns (loss, NLL per token)."""
        K = chosen.shape[1]
        active = torch.arange(K, device=u.device)[None, :] < depth[:, None]
        a = alpha * active
        theta = a / a.sum(1, keepdim=True)
        p = torch.einsum("bk,bkv->bv", theta, self.phi()[chosen]).clamp_min(1e-12)
        nll = -(counts * torch.log(p)).sum() / counts.sum()
        u_hat = torch.einsum("bk,bkd->bd", a, self.unit_codes()[chosen])
        commit = (u - u_hat).pow(2).sum(1).mean()
        return nll + commit_weight * commit, nll

    @torch.no_grad()
    def transform(self, counts, k=None):
        """Topic sequence of new documents: (chosen, theta) for the first k topics (default k_max = min(10, T))."""
        k = k or min(10, self.codes.shape[0])
        was = self.training
        self.eval()
        chosen, alpha = self.pursuit(self.encode(counts), k)
        self.train(was)
        return chosen, alpha / alpha.sum(1, keepdim=True)
