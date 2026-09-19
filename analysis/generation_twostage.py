"""Two-stage generation: node sequence (stage A) + explicit edge completion (stage B).

Stage A generates node features autoregressively (feature-only VQ tokens).
Stage B serializes the upper-triangular adjacency into fixed-size edge chunks, quantizes
them, and generates the chunk sequence autoregressively conditioned on the node-feature
summary. Because stage B predicts edges between *all* node pairs, triangles/rings can form.
"""
import json
import statistics
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from model.vector_quantizer import VectorQuantizer

from .dataset import load_dataset, split_indices
from .metrics import summarize_graphs
from .prior import make_prior, train_prior, sample_tokens

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "twostage.yaml"
RESULTS_DIR = ROOT / "results"


# ---------------------------------------------------------------- models
class FeatureTokenizer(nn.Module):
    """Stage A: quantize a node's one-hot feature into a discrete token."""

    def __init__(self, in_dim, hidden, codebook_size):
        super().__init__()
        self.in_dim = in_dim
        self.encoder = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.quantizer = VectorQuantizer(codebook_size, hidden)
        self.decoder = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, in_dim))

    def forward(self, x):
        z = self.encoder(x)
        z_q, idx, vq = self.quantizer(z)
        out = self.decoder(z_q)
        return out, idx, vq


class EdgeTokenizer(nn.Module):
    """Stage B: quantize an edge chunk (B bits) into a discrete token."""

    def __init__(self, B, hidden, codebook_size):
        super().__init__()
        self.B = B
        self.encoder = nn.Sequential(nn.Linear(B, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.quantizer = VectorQuantizer(codebook_size, hidden)
        self.decoder = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, B))

    def forward(self, c):
        z = self.encoder(c)
        z_q, idx, vq = self.quantizer(z)
        out = self.decoder(z_q)
        return out, idx, vq


class PairEdgeModel(nn.Module):
    """Stage B v3 (P0): autoregressive Bernoulli edge model with per-chunk
    pair conditioning (reviewer 2.3):

        s_q   = GRU(s_{q-1}, emb(c_{q-1}) + Pool({proj(r_ij) : (i,j) in chunk q}))
        p(c_q) = prod_b Bernoulli(c_{q,b}; sigmoid(w_b^T s_q))

    ``B`` bits per chunk, teacher forcing on chunk bits during training, and a
    per-bit padding mask so partial chunks are handled exactly.
    """

    def __init__(self, B: int, pair_dim: int, hidden: int, num_layers: int = 2,
                 within_ar: bool = False, per_pair: bool = False):
        super().__init__()
        self.B = B
        self.hidden = hidden
        self.within_ar = within_ar
        self.per_pair = per_pair
        self.start = nn.Parameter(torch.zeros(1, hidden))
        self.pair_proj = nn.Sequential(nn.Linear(pair_dim, hidden), nn.ReLU(),
                                       nn.Linear(hidden, hidden))
        self.in_proj = nn.Linear(B, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=num_layers, batch_first=True)
        if within_ar:
            self.inner = nn.GRU(hidden + 1, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)
        elif per_pair:
            # B3 control: readout of bit b from [chunk state s_q ; its own pair feature]
            self.head = nn.Linear(2 * hidden, 1)
        else:
            self.head = nn.Linear(hidden, B)

    def _pool(self, pair_feats, mask):
        """Masked mean-pool of projected per-bit pair features within each chunk."""
        if pair_feats.dim() == 3:  # (L, B, pair_dim)
            pair_feats = pair_feats.unsqueeze(0)
            mask = mask.unsqueeze(0)
        p = self.pair_proj(pair_feats)                       # (N, L, B, hidden)
        denom = mask.sum(-1, keepdim=True).clamp(min=1.0)    # (N, L, 1)
        return (p * mask.unsqueeze(-1)).sum(2) / denom       # (N, L, hidden)

    def _outer(self, chunk_bits, pair_feats, mask):
        emb = self.in_proj(chunk_bits)
        prev = torch.cat([self.start.expand(emb.size(0), 1, -1), emb[:, :-1, :]], dim=1)
        inp = prev + self._pool(pair_feats, mask)
        out, _ = self.gru(inp)
        return out

    def forward(self, chunk_bits, pair_feats, mask):
        """chunk_bits (N,L,B), pair_feats (N,L,B,pair_dim), mask (N,L,B)."""
        s = self._outer(chunk_bits, pair_feats, mask)
        if not self.within_ar and not self.per_pair:
            return self.head(s)
        if self.per_pair:
            N, L, B = chunk_bits.shape
            p = self.pair_proj(pair_feats)                          # (N,L,B,H)
            cat = torch.cat([s.unsqueeze(2).expand(-1, -1, B, -1), p], -1)  # (N,L,B,2H)
            return self.head(cat).squeeze(-1)                       # (N,L,B)
        N, L, B = chunk_bits.shape
        bits_in = torch.cat([torch.zeros(N, L, 1, device=chunk_bits.device),
                             chunk_bits[..., :-1]], -1)          # shifted prev bit
        inner_in = torch.cat([s.unsqueeze(2).expand(-1, -1, B, -1),
                              bits_in.unsqueeze(-1)], -1)        # (N,L,B,H+1)
        out, _ = self.inner(inner_in.reshape(N * L, B, -1))
        return self.head(out.reshape(N, L, B, -1)).squeeze(-1)   # (N,L,B)


def _pos_enc(i, j, n):
    """Deterministic positional features of pair (i, j) in the upper-triangle order."""
    denom = max(n, 1)
    return np.array([np.sin(2 * np.pi * i / denom), np.cos(2 * np.pi * i / denom),
                     np.sin(2 * np.pi * j / denom), np.cos(2 * np.pi * j / denom),
                     (j - i) / denom], dtype=np.float32)


def build_pair_chunks(tok_embs, feats, n, B):
    """Serialise all upper-triangular pairs into B-bit chunks with per-pair features.

    r_ij = [h_i || h_j || h_i*h_j || |h_i-h_j| || pos(i,j) || g], where
    h_i = [codebook token embedding of node i || one-hot feature of node i] and
    g is the graph-level mean token. Padding bits of the last chunk carry zero
    features and mask 0.

    Returns (pair_feats (L,B,pair_dim), mask (L,B), pair_dim).
    """
    H = tok_embs.shape[1]
    in_dim = feats.shape[1]
    h = np.concatenate([tok_embs, feats], axis=1)   # (n, H+in_dim)
    g = h.mean(0)
    pair_dim = 5 * (H + in_dim) + 5
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            r = np.concatenate([h[i], h[j], h[i] * h[j], np.abs(h[i] - h[j]),
                                _pos_enc(i, j, n), g])
            pairs.append(r)
    if not pairs:  # n < 2: degenerate all-masked chunk
        pairs = [np.zeros(pair_dim, dtype=np.float32)] * B
    n_pairs = len(pairs)
    pad = (-n_pairs) % B
    for _ in range(pad):
        pairs.append(np.zeros(pair_dim, dtype=np.float32))
    L = len(pairs) // B
    pf = np.asarray(pairs, dtype=np.float32).reshape(L, B, pair_dim)
    mask = np.zeros((L, B), dtype=np.float32)
    mask.reshape(-1)[:n_pairs] = 1.0
    return pf, mask, pair_dim


def corrupt_embs(embs, codebook, prob):
    """Random token replacement on node embeddings (scheduled sampling, reviewer 2.4)."""
    if prob <= 0:
        return embs
    n = embs.shape[0]
    mask = torch.rand(n, device=embs.device) < prob
    if mask.any():
        idx = torch.randint(0, codebook.size(0), (int(mask.sum().item()),), device=embs.device)
        out = embs.clone()
        out[mask] = codebook[idx]
        return out
    return embs


def train_pair_model(model, data, epochs, lr, device, bs=32, pos_weight=None):
    """Teacher-forced training of :class:`PairEdgeModel` with masked BCE.

    ``data``: list of (chunk_bits (L,B), pair_feats (L,B,pair_dim), mask (L,B)) tensors.
    ``pos_weight``: optional per-positive-bit BCE weight (edge-imbalance calibration).
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        perm = torch.randperm(len(data), device=device)
        for s in range(0, len(data), bs):
            batch = [data[i] for i in perm[s:s + bs].tolist()]
            L = max(b[0].shape[0] for b in batch)
            B = model.B
            bits = torch.zeros(len(batch), L, B, device=device)
            pf = torch.zeros(len(batch), L, *batch[0][1].shape[1:], device=device)
            mask = torch.zeros(len(batch), L, B, device=device)
            for k, (b, p, m) in enumerate(batch):
                bits[k, :b.shape[0]] = b
                pf[k, :p.shape[0]] = p
                mask[k, :m.shape[0]] = m
            opt.zero_grad()
            logits = model(bits, pf, mask)
            w = mask
            if pos_weight is not None:
                w = mask * (1.0 + (pos_weight - 1.0) * bits)
            loss = F.binary_cross_entropy_with_logits(
                logits, bits, weight=w, reduction="sum") / mask.sum().clamp(min=1)
            loss.backward()
            opt.step()
    return model


def sample_pair_model(model, pair_feats, mask, device, temperature=1.0):
    """Autoregressively sample chunk bits for one graph (masked bits forced to 0)."""
    model.eval()
    L = pair_feats.shape[0]
    h = None
    prev = model.start
    chunks = []
    with torch.no_grad():
        for q in range(L):
            pooled = model._pool(pair_feats[q:q + 1], mask[q:q + 1])  # (1,1,hidden)
            inp = prev.unsqueeze(1) + pooled                         # (1,1,hidden)
            out, h = model.gru(inp, h)
            s_q = out[0, 0]                                          # (hidden,)
            if model.per_pair:
                p = model.pair_proj(pair_feats[q:q + 1])[0]           # (B, hidden)
                cat = torch.cat([s_q.unsqueeze(0).expand(model.B, -1), p], -1)
                logits = model.head(cat).squeeze(-1) / temperature    # (B,)
                p = torch.sigmoid(logits)
                bits = (torch.rand_like(p) < p).float()
            elif not model.within_ar:
                logits = model.head(out)[0, 0] / temperature
                p = torch.sigmoid(logits)
                bits = (torch.rand_like(p) < p).float()
            else:
                bits_list = []
                ih = None
                bit_prev = torch.zeros(1, device=device)
                for b in range(model.B):
                    inner_in = torch.cat([s_q, bit_prev], -1).view(1, 1, -1)
                    io, ih = model.inner(inner_in, ih)
                    p = torch.sigmoid(model.head(io)[0, 0, 0] / temperature)
                    bit = (torch.rand(1, device=device) < p).float()
                    bits_list.append(bit)
                    bit_prev = bit
                bits = torch.stack(bits_list).squeeze(-1)
            bits[mask[q] == 0] = 0.0
            chunks.append(bits.cpu().numpy())
            prev = model.in_proj(bits.unsqueeze(0))
    return np.array(chunks, dtype=np.float32)


# ---------------------------------------------------------------- utils
def _pair_chunks_of(ftok, graphs, B, pair_cond, device):
    """Oracle pair chunks (no corruption) of a graph list, for eval/calibration."""
    out = []
    with torch.no_grad():
        for (a, x, _) in graphs:
            _, idx, _ = ftok(torch.from_numpy(x).to(device))
            embs = ftok.quantizer.codebook[idx]
            pf, mask, _ = build_pair_chunks(embs.cpu().numpy(), x, a.shape[0], B)
            if pair_cond == "none":
                pf = np.zeros_like(pf)
            out.append((torch.from_numpy(serialize_edges(a, B)).to(device),
                        torch.from_numpy(pf).to(device),
                        torch.from_numpy(mask).to(device)))
    return out


def _calibrate_temperature(pair_model, ftok, val_graphs, B, pair_cond, device):
    """Grid-search tau minimizing the unweighted masked BCE on the validation split.

    B1: with the unweighted probability target, temperature calibration on the
    validation set is a legitimate calibration step (it optimizes the same
    unweighted NLL objective), not a density re-targeting hack.
    """
    data = _pair_chunks_of(ftok, val_graphs, B, pair_cond, device)
    pair_model.eval()
    best_tau, best_loss = 1.0, float("inf")
    with torch.no_grad():
        for tau in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0]:
            tot = den = 0.0
            for bits, pf, mask in data:
                logits = pair_model(bits.unsqueeze(0), pf.unsqueeze(0),
                                    mask.unsqueeze(0))[0] / tau
                tot += F.binary_cross_entropy_with_logits(
                    logits, bits, weight=mask, reduction="sum").item()
                den += mask.sum().item()
            avg = tot / max(den, 1)
            if avg < best_loss:
                best_loss, best_tau = avg, tau
    return best_tau


def _ece_np(p, y, m, n_bins=10):
    p = p.detach().cpu().numpy().ravel()
    y = y.cpu().numpy().ravel()
    m = m.cpu().numpy().ravel()
    sel = m > 0.5
    p, y = p[sel], y[sel]
    if len(p) == 0:
        return 0.0
    order = np.argsort(p)
    p, y = p[order], y[order]
    ece = 0.0
    for k in range(n_bins):
        lo, hi = int(k * len(p) / n_bins), int((k + 1) * len(p) / n_bins)
        if hi <= lo:
            continue
        ece += (hi - lo) / len(p) * abs(p[lo:hi].mean() - y[lo:hi].mean())
    return float(ece)


def _conditional_metrics(pair_model, ftok, test_graphs, B, device):
    """Unweighted conditional NLL/Brier/ECE (pos/neg) under TRUE conditions (B5)."""
    data = _pair_chunks_of(ftok, test_graphs, B, "pair", device)
    pair_model.eval()
    agg = {lab: {"nll": 0.0, "brier": 0.0, "n": 0} for lab in ("positive", "negative")}
    ece_parts = {"positive": [], "negative": []}
    with torch.no_grad():
        for bits, pf, mask in data:
            logits = pair_model(bits.unsqueeze(0), pf.unsqueeze(0), mask.unsqueeze(0))[0]
            p = torch.sigmoid(logits)
            for lab, sel in (("positive", bits > 0.5), ("negative", bits < 0.5)):
                m = mask * sel.float()
                den = m.sum().item()
                if den == 0:
                    continue
                bce = -(bits * torch.log(p + 1e-12) + (1 - bits) * torch.log(1 - p + 1e-12))
                agg[lab]["nll"] += (bce * m).sum().item()
                agg[lab]["brier"] += (((p - bits) ** 2) * m).sum().item()
                agg[lab]["n"] += den
                ece_parts[lab].append(_ece_np(p, bits, m))
    out = {}
    for lab in ("positive", "negative"):
        den = agg[lab]["n"]
        out[lab] = {
            "nll_nats_per_bit": round(agg[lab]["nll"] / max(den, 1), 5),
            "brier": round(agg[lab]["brier"] / max(den, 1), 5),
            "ece": round(float(np.mean(ece_parts[lab])), 5) if ece_parts[lab] else None,
            "n_bits": int(den),
        }
    return out


def build_cond(ftok, feats, cond_type, device):
    """Stage-B conditioning summary of a node-feature set (reviewer Q2 variants).

    ``mean`` (baseline): per-class proportions of the one-hot features.
    ``mean_std``: mean + per-dimension standard deviation (feature variance).
    ``emb_pool``: mean-pooled *quantized* feature-token embeddings (a learned
    graph-level summary, as opposed to the raw feature statistics).
    ``feats`` is either a numpy feature matrix (training) or a list of one-hot
    arrays (generation); ``ftok`` must already be trained for ``emb_pool``.
    """
    in_dim = ftok.in_dim
    if len(feats) == 0:
        if cond_type == "emb_pool":
            return torch.zeros(ftok.quantizer.codebook.size(1), device=device)
        return torch.zeros(in_dim if cond_type == "mean" else 2 * in_dim, device=device)
    x = torch.tensor(np.asarray(feats, dtype=np.float32), device=device)
    if cond_type == "mean_std":
        # unbiased=False: population std is well-defined for single-node graphs
        # (unbiased std with n=1 yields NaN, which crashes downstream sampling).
        return torch.cat([x.mean(0), torch.nan_to_num(torch.std(x, dim=0, unbiased=False))])
    if cond_type == "emb_pool":
        with torch.no_grad():
            _, idx, _ = ftok(x)
            emb = ftok.quantizer.codebook[idx]
        return emb.mean(0)
    return x.mean(0)


COND_DIM = {"mean": 1, "mean_std": 2, "emb_pool": 0}  # multiplier; emb_pool dim = hidden


def serialize_edges(adj, B):
    n = adj.shape[0]
    bits = []
    for i in range(n):
        for j in range(i + 1, n):
            bits.append(adj[i, j])
    while len(bits) % B != 0:
        bits.append(0.0)
    chunks = [bits[k:k + B] for k in range(0, len(bits), B)]
    if not chunks:  # n < 2 -> degenerate single chunk of zeros
        chunks = [[0.0] * B]
    return np.array(chunks, dtype=np.float32)


def deserialize_edges(chunks, n):
    bits = np.asarray(chunks).reshape(-1)
    adj = np.zeros((n, n), dtype=np.float32)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if idx < len(bits):
                adj[i, j] = adj[j, i] = bits[idx]
            idx += 1
    return adj


# ---------------------------------------------------------------- training
def train_tokenizer(tok, xs, loss_fn, epochs, lr, beta, device, bs=256):
    opt = torch.optim.Adam(tok.parameters(), lr=lr)
    x = torch.from_numpy(np.concatenate(xs, 0)).to(device)
    n = x.size(0)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            opt.zero_grad()
            out, _, vq = tok(x[idx])
            loss = loss_fn(out, x[idx]) + beta * vq
            loss.backward()
            opt.step()
    return tok


def evaluate_generated(gen_graphs, train_graphs, signature="wl"):
    """Compute generation metrics on already-built generated graphs."""
    return summarize_graphs(gen_graphs, train_graphs, include_hist_and_conn=False,
                            signature=signature)


def run_one(dataset: str, cfg: dict, device: str, collect_graphs: bool = False,
            graphs: list = None, val_graphs: list = None, test_graphs: list = None):
    """Train and evaluate the two-stage generator once (single seed, from cfg['seed']).

    ``cond_type`` selects the Stage-B conditioning: ``mean`` / ``mean_std`` /
    ``emb_pool`` (global summaries, reviewer Q2) or ``pair`` (pair-conditioned
    edge model, P0/P1). ``graphs`` optionally overrides the dataset (the 80% train
    split). Batch-2 options: ``pair_cond=none`` (B2b unconditional control),
    ``pair_readout=per_pair`` (B3 per-pair readout control),
    ``stage_b_temp_calib=true`` with ``val_graphs`` (B1 validation calibration),
    ``test_graphs`` (B5 conditional metrics under true conditions), and
    ``dump_samples`` (sparse sample dump for held-out reference evaluation).
    With ``collect_graphs=True`` the generated (adj, feat) list is returned as a
    second element.
    """
    B = cfg["edge_chunk_bits"]
    cond_type = cfg.get("cond_type", "mean")
    prior_type = cfg.get("prior_type", "gru")
    prior_hidden = cfg.get("prior_hidden", cfg["hidden"])
    if graphs is None:
        graphs = load_dataset(dataset, cfg.get("n_qm9", 5000))
    in_dim = graphs[0][1].shape[1]
    node_counts = [a.shape[0] for a, _, _ in graphs]

    # ---- Stage A: feature tokens
    feats = [x for _, x, _ in graphs]
    ftok = FeatureTokenizer(in_dim, cfg["hidden"], cfg["codebook_size"]).to(device)
    ftok = train_tokenizer(ftok, feats, lambda o, t: F.mse_loss(o, t),
                           cfg["stage_a_epochs"], cfg["lr"], cfg["commitment_beta"], device)
    vocab = cfg["codebook_size"] + 2
    sos, eos = cfg["codebook_size"], cfg["codebook_size"] + 1
    f_tokens = []
    for x in feats:
        xt = torch.from_numpy(x).to(device)
        _, idx, _ = ftok(xt)
        f_tokens.append(idx.tolist())
    f_prior = make_prior(vocab, prior_hidden, 0, prior_type).to(device)
    f_prior = train_prior(f_prior, f_tokens, cfg["stage_a_epochs"], cfg["lr"], device, sos, eos)

    if cond_type == "pair":
        # ---- Stage B v3: pair-conditioned autoregressive edge model (P0/P1)
        corrupt = cfg.get("corrupt_prob", 0.15)
        pair_cond = cfg.get("pair_cond", "pair")            # "pair" | "none" (B2b)
        per_pair = cfg.get("pair_readout", "pooled") == "per_pair"  # B3 control
        data = []
        with torch.no_grad():
            for (a, x, _) in graphs:
                xt = torch.from_numpy(x).to(device)
                _, idx, _ = ftok(xt)
                embs = corrupt_embs(ftok.quantizer.codebook[idx],
                                    ftok.quantizer.codebook, corrupt)
                pf, mask, pair_dim = build_pair_chunks(embs.cpu().numpy(), x, a.shape[0], B)
                if pair_cond == "none":
                    pf = np.zeros_like(pf)
                data.append((torch.from_numpy(serialize_edges(a, B)).to(device),
                             torch.from_numpy(pf).to(device),
                             torch.from_numpy(mask).to(device)))
        pair_model = PairEdgeModel(B, pair_dim, cfg.get("pair_hidden", cfg["hidden"]),
                                   within_ar=cfg.get("pair_within_ar", False),
                                   per_pair=per_pair).to(device)
        pair_model = train_pair_model(pair_model, data, cfg["stage_b_epochs"], cfg["lr"],
                                      device, pos_weight=cfg.get("pair_pos_weight", None))

        # ---- B1: validation-set temperature calibration of the unweighted target
        temperature = cfg.get("temperature", 1.0)
        if cfg.get("stage_b_temp_calib", False) and val_graphs:
            temperature = _calibrate_temperature(pair_model, ftok, val_graphs, B,
                                                 pair_cond, device)

        # ---- B5: held-out conditional metrics under TRUE conditions
        cond_test = None
        if test_graphs:
            cond_test = _conditional_metrics(pair_model, ftok, test_graphs, B, device)

        # ---- Generate two-stage (pair condition)
        ftok.eval(); f_prior.eval(); pair_model.eval()
        generated = []
        counts = np.random.choice(node_counts, cfg["num_samples"])
        with torch.no_grad():
            for cnt in counts:
                cnt = int(min(cnt, cfg.get("max_nodes", 128)))
                ft = sample_tokens(f_prior, cnt, None, device, sos, eos,
                                   cfg.get("temperature", 1.0))
                if not ft:
                    continue
                feats = []
                for t in ft:
                    z = ftok.quantizer.codebook[t]
                    out = ftok.decoder(z)
                    f = torch.zeros(in_dim, device=device)
                    f[out.argmax()] = 1.0
                    feats.append(f.cpu().numpy())
                n = len(feats)
                embs = ftok.quantizer.codebook[torch.tensor(ft, device=device)]
                pf, mask, _ = build_pair_chunks(embs.cpu().numpy(),
                                                np.array(feats, dtype=np.float32), n, B)
                if pair_cond == "none":
                    pf = np.zeros_like(pf)
                chunks = sample_pair_model(pair_model, torch.from_numpy(pf).to(device),
                                           torch.from_numpy(mask).to(device), device,
                                           temperature)
                adj = deserialize_edges(chunks, n)
                generated.append((feats, None, adj))
    else:
        # ---- Stage B (global summary condition): edge chunks via VQ tokens
        edge_chunks = []
        conds = []
        for (a, x, _) in graphs:
            edge_chunks.append(serialize_edges(a, B))
            conds.append(build_cond(ftok, x, cond_type, device))
        etok = EdgeTokenizer(B, cfg["hidden"], cfg["codebook_size"]).to(device)
        etok = train_tokenizer(etok, edge_chunks, lambda o, t: F.binary_cross_entropy_with_logits(o, t),
                               cfg["stage_b_epochs"], cfg["lr"], cfg["commitment_beta"], device)
        e_tokens = []
        for c in edge_chunks:
            ct = torch.from_numpy(c).to(device)
            _, idx, _ = etok(ct)
            e_tokens.append(idx.tolist())
        cond_dim = in_dim * COND_DIM[cond_type] if cond_type != "emb_pool" else cfg["hidden"]
        e_prior = make_prior(vocab, prior_hidden, cond_dim, prior_type).to(device)
        e_prior = train_prior(e_prior, e_tokens, cfg["stage_b_epochs"], cfg["lr"], device, sos, eos, conds=conds)

        # ---- Generate two-stage
        ftok.eval(); etok.eval(); f_prior.eval(); e_prior.eval()
        generated = []
        counts = np.random.choice(node_counts, cfg["num_samples"])
        with torch.no_grad():
            for cnt in counts:
                cnt = int(min(cnt, cfg.get("max_nodes", 128)))
                # stage A: node features
                ft = sample_tokens(f_prior, cnt, None, device, sos, eos, cfg.get("temperature", 1.0))
                feats = []
                for t in ft:
                    z = ftok.quantizer.codebook[t]
                    out = ftok.decoder(z)
                    f = torch.zeros(in_dim, device=device)
                    f[out.argmax()] = 1.0
                    feats.append(f.cpu().numpy())
                n = len(feats)
                # stage B: edges (condition on the node-feature summary)
                cond = build_cond(ftok, feats, cond_type, device)
                n_chunks = max(1, (n * (n - 1) // 2 + B - 1) // B)
                et = sample_tokens(e_prior, n_chunks, cond, device, sos, eos,
                                   cfg.get("temperature", 1.0), mask_eos=True, stop_on_eos=False)
                chunks = []
                for t in et:
                    z = etok.quantizer.codebook[t]
                    out = etok.decoder(z)
                    p = torch.sigmoid(out)
                    chunks.append((torch.rand_like(p) < p).float().cpu().numpy())
                if chunks and n > 0:
                    adj = deserialize_edges(np.array(chunks), n)
                    generated.append((feats, None, adj))
    # wrap into the format evaluate() expects: list of (adj, feat) for gen, and reuse train graphs
    gen_graphs = [(adj, np.array(feats, dtype=np.float32)) for (feats, _, adj) in generated if len(feats) > 0]
    if cfg.get("dump_samples"):
        # B5: sparse dump of generated graphs for held-out reference evaluation
        import os
        dump_dir = Path(os.environ.get("DUMP_DIR", RESULTS_DIR)) / "batch2_samples"
        dump_dir.mkdir(parents=True, exist_ok=True)
        rows, cols, ns, feats_l = [], [], [], []
        for adj, f in gen_graphs:
            r, c = np.triu(adj, 1).nonzero()
            rows.append(r.astype(np.int16))
            cols.append(c.astype(np.int16))
            ns.append(int(adj.shape[0]))
            feats_l.append(np.argmax(f, 1).astype(np.int8))
        tag = cfg.get("dump_tag", "gen")
        np.savez_compressed(
            dump_dir / f"{tag}_{dataset}_seed{cfg.get('seed', 0)}.npz",
            g_rows=np.asarray(rows, dtype=object), g_cols=np.asarray(cols, dtype=object),
            g_n=np.asarray(ns, dtype=np.int32), g_feat=np.asarray(feats_l, dtype=object),
            allow_pickle=True)
    metrics = evaluate_generated(gen_graphs, graphs,
                                 signature={"QM9": "qm9", "MUTAG": "isotope"}.get(dataset, "wl"))
    if cond_type == "pair":
        metrics["conditional_test"] = cond_test
        metrics["stage_b_temperature"] = temperature
    if collect_graphs:
        return metrics, gen_graphs
    return metrics


def run_gap(dataset: str, cfg: dict, device: str, seeds=(0,)) -> dict:
    """Three-condition edge-NLL diagnosis (reviewer 2.4): oracle / reconstructed /
    generated.

    Train on an 80% split, then on each held-out graph compare the Stage-B edge
    NLL when the pair features come from (a) tokens of the *true* node features,
    (b) tokens of the deterministic Stage-A reconstruction, and (c) tokens of one
    Stage-A prior sample. All three conditions are evaluated on the subgraph
    induced by the first ``m`` nodes, where ``m`` is the length of the Stage-A
    sample, so the comparison is paired per graph.
    """
    graphs = load_dataset(dataset, cfg.get("n_qm9", 5000))
    in_dim = graphs[0][1].shape[1]
    B = cfg["edge_chunk_bits"]
    split = tuple(cfg.get("split", [0.8, 0.1, 0.1]))
    out = {"dataset": dataset, "config": {"corrupt_prob": cfg.get("corrupt_prob", 0.15),
                                          "B": B}, "seeds": {}}
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        tr_idx, va_idx, te_idx = split_indices(len(graphs), seed, split)
        train = [graphs[i] for i in tr_idx]
        # ---- Stage A on train split
        feats = [x for _, x, _ in train]
        ftok = FeatureTokenizer(in_dim, cfg["hidden"], cfg["codebook_size"]).to(device)
        ftok = train_tokenizer(ftok, feats, lambda o, t: F.mse_loss(o, t),
                               cfg["stage_a_epochs"], cfg["lr"], cfg["commitment_beta"], device)
        vocab = cfg["codebook_size"] + 2
        sos, eos = cfg["codebook_size"], cfg["codebook_size"] + 1
        f_tokens = []
        for x in feats:
            xt = torch.from_numpy(x).to(device)
            with torch.no_grad():
                _, idx, _ = ftok(xt)
            f_tokens.append(idx.tolist())
        f_prior = make_prior(vocab, cfg.get("prior_hidden", cfg["hidden"]), 0,
                             cfg.get("prior_type", "gru")).to(device)
        f_prior = train_prior(f_prior, f_tokens, cfg["stage_a_epochs"], cfg["lr"], device, sos, eos)
        # ---- Stage B pair model on train split (oracle + corruption)
        data = []
        with torch.no_grad():
            for (a, x, _) in train:
                xt = torch.from_numpy(x).to(device)
                _, idx, _ = ftok(xt)
                embs = corrupt_embs(ftok.quantizer.codebook[idx],
                                    ftok.quantizer.codebook, cfg.get("corrupt_prob", 0.15))
                pf, mask, pair_dim = build_pair_chunks(embs.cpu().numpy(), x, a.shape[0], B)
                data.append((torch.from_numpy(serialize_edges(a, B)).to(device),
                             torch.from_numpy(pf).to(device),
                             torch.from_numpy(mask).to(device)))
        pair_model = PairEdgeModel(B, pair_dim, cfg.get("pair_hidden", cfg["hidden"]),
                                   within_ar=cfg.get("pair_within_ar", False)).to(device)
        pair_model = train_pair_model(pair_model, data, cfg["stage_b_epochs"], cfg["lr"],
                                      device, pos_weight=cfg.get("pair_pos_weight", None))
        ftok.eval(); f_prior.eval(); pair_model.eval()
        # ---- evaluate three conditions on val + test
        sums = {"oracle": 0.0, "reconstructed": 0.0, "generated": 0.0}
        n_bits = {"oracle": 0, "reconstructed": 0, "generated": 0}
        n_graphs = 0
        with torch.no_grad():
            for i in list(va_idx) + list(te_idx):
                a, x, _ = graphs[i]
                n = a.shape[0]
                # generated condition: one Stage-A sample defines m and its tokens
                ft = sample_tokens(f_prior, n, None, device, sos, eos,
                                   cfg.get("temperature", 1.0))
                m = len(ft)
                if m < 2:
                    continue
                xg = np.zeros((m, in_dim), dtype=np.float32)
                for k, t in enumerate(ft):
                    dec = ftok.decoder(ftok.quantizer.codebook[t])
                    xg[k, int(dec.argmax())] = 1.0
                sub_a = a[:m, :m]
                sub_x = x[:m]
                bits_true = torch.from_numpy(serialize_edges(sub_a, B)).to(device)
                # oracle
                xt = torch.from_numpy(sub_x).to(device)
                _, idx_o, _ = ftok(xt)
                pf_o, mk_o, _ = build_pair_chunks(ftok.quantizer.codebook[idx_o].cpu().numpy(),
                                                  sub_x, m, B)
                # reconstructed
                dec = ftok.decoder(ftok.quantizer.codebook[idx_o])
                xr = torch.zeros_like(xt)
                xr[torch.arange(m, device=device), dec.argmax(1)] = 1.0
                with torch.no_grad():
                    _, idx_r, _ = ftok(xr)
                pf_r, mk_r, _ = build_pair_chunks(ftok.quantizer.codebook[idx_r].cpu().numpy(),
                                                  xr.cpu().numpy(), m, B)
                # generated
                pf_g, mk_g, _ = build_pair_chunks(
                    ftok.quantizer.codebook[torch.tensor(ft, device=device)].cpu().numpy(),
                    xg, m, B)
                for name, (pfx, mkx) in {"oracle": (pf_o, mk_o),
                                         "reconstructed": (pf_r, mk_r),
                                         "generated": (pf_g, mk_g)}.items():
                    pfT = torch.from_numpy(pfx).to(device).unsqueeze(0)
                    mkT = torch.from_numpy(mkx).to(device).unsqueeze(0)
                    logits = pair_model(bits_true.unsqueeze(0), pfT, mkT)
                    bce = F.binary_cross_entropy_with_logits(
                        logits, bits_true.unsqueeze(0), weight=mkT, reduction="sum")
                    sums[name] += bce.item()
                    n_bits[name] += int(mkT.sum().item())
                n_graphs += 1
        per = {name: round(sums[name] / max(n_bits[name], 1), 6) for name in sums}
        per["n_graphs"] = n_graphs
        out["seeds"][str(seed)] = per
        print(f"[gap/{dataset}/seed{seed}]", json.dumps(per), flush=True)
    return out


def _aggregate(per_seed: dict) -> dict:
    """Fold a {seed: metrics-dict} mapping into {mean-dict, std-dict} (same shape).

    Non-numeric leaves (e.g. the ``signature`` string) are carried through unchanged.
    """
    seeds = sorted(per_seed.keys())
    first = per_seed[seeds[0]]

    def fold(d, path):
        out_mean, out_std = {}, {}
        for k, v in d.items():
            if isinstance(v, dict):
                m, s = fold(v, path + [k])
                out_mean[k], out_std[k] = m, s
            else:
                vals = [per_seed[sd] for sd in seeds]
                for p in path:
                    vals = [x[p] for x in vals]
                vals = [x[k] for x in vals]
                if all(isinstance(x, (int, float)) for x in vals):
                    out_mean[k] = statistics.mean(vals)
                    out_std[k] = statistics.stdev(vals) if len(vals) > 1 else 0.0
                else:
                    out_mean[k], out_std[k] = vals[0], None
        return out_mean, out_std

    mean_d, std_d = fold(first, [])
    return {"metrics": mean_d, "std": std_d, "seeds": seeds}


def run() -> dict:
    cfg = yaml.safe_load(open(CONFIG_PATH, encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = cfg.get("seeds", [cfg.get("seed", 0)])
    B = cfg["edge_chunk_bits"]
    seeds_dir = RESULTS_DIR / "twostage_seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for ds in cfg["datasets"]:
        print(f"\n=== {ds} ===", flush=True)
        per_seed = {}
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            print(f"  seed {seed}...", flush=True)
            m = run_one(ds, dict(cfg, seed=seed), device)
            per_seed[str(seed)] = m
            json.dump(m, open(seeds_dir / f"{ds}_{seed}.json", "w"), indent=2)
            print(json.dumps(m, indent=2), flush=True)
        agg = _aggregate(per_seed)
        results[ds] = agg
        print(f"[{ds}] aggregated:", json.dumps(agg["metrics"], indent=2), flush=True)

    json.dump(results, open(RESULTS_DIR / "twostage_results.json", "w"), indent=2)
    return results


def run_prior_ablation() -> dict:
    """Reviewer Q1 (two-stage): transformer-128 prior for both Stage A and Stage B
    vs. the GRU-32 baseline already in twostage_seeds. MUTAG + QM9, three seeds."""
    cfg = yaml.safe_load(open(CONFIG_PATH, encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = cfg.get("seeds", [0, 1, 2])
    ov = {"prior_type": "transformer", "prior_hidden": 128}
    seeds_dir = RESULTS_DIR / "twostage_seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)
    out = {"baseline": "gru32 (twostage_seeds)", "variant": "tf128"}
    for ds in ["MUTAG", "QM9"]:
        print(f"\n=== prior tf128 / {ds} ===", flush=True)
        per_seed = {}
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            m = run_one(ds, dict(cfg, seed=seed, **ov), device)
            per_seed[str(seed)] = m
            json.dump(m, open(seeds_dir / f"prior_tf128_{ds}_{seed}.json", "w"), indent=2)
            print(f"[tf128/{ds}/seed{seed}]", json.dumps(m), flush=True)
        out[ds] = _aggregate(per_seed)
        json.dump(out, open(RESULTS_DIR / "prior_ablation_twostage.json", "w"), indent=2)
    return out


def run_cond_ablation() -> dict:
    """Reviewer Q2: Stage-B conditioning variants (mean baseline in twostage_seeds):
    mean_std (mean + feature variance) and emb_pool (learned graph-level summary)."""
    cfg = yaml.safe_load(open(CONFIG_PATH, encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = cfg.get("seeds", [0, 1, 2])
    variants = [
        ("mean_std", {"cond_type": "mean_std"}),
        ("emb_pool", {"cond_type": "emb_pool"}),
    ]
    seeds_dir = RESULTS_DIR / "twostage_seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)
    out = {"baseline": "mean (twostage_seeds)", "variants": {}}
    for name, ov in variants:
        out["variants"][name] = {}
        for ds in ["MUTAG", "QM9"]:
            print(f"\n=== cond {name} / {ds} ===", flush=True)
            per_seed = {}
            for seed in seeds:
                torch.manual_seed(seed); np.random.seed(seed)
                m = run_one(ds, dict(cfg, seed=seed, **ov), device)
                per_seed[str(seed)] = m
                json.dump(m, open(seeds_dir / f"cond_{name}_{ds}_{seed}.json", "w"), indent=2)
                print(f"[{name}/{ds}/seed{seed}]", json.dumps(m), flush=True)
            out["variants"][name][ds] = _aggregate(per_seed)
            json.dump(out, open(RESULTS_DIR / "cond_ablation.json", "w"), indent=2)
    return out


def run_sensitivity() -> dict:
    """E9: edge-chunk size B sweep for the two-stage generator (QM9, seed 0)."""
    cfg = yaml.safe_load(open(CONFIG_PATH, encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = "QM9"
    keys = ["validity", "mean_degree", "triangle"]
    out = {"dataset": dataset, "sweeps": {"edge_chunk_bits": {}}}
    for B in [4, 8, 16]:
        torch.manual_seed(0); np.random.seed(0)
        print(f"\n=== edge_chunk_bits={B} ===", flush=True)
        m = run_one(dataset, dict(cfg, seed=0, edge_chunk_bits=B), device)
        out["sweeps"]["edge_chunk_bits"][str(B)] = {k: m[k] for k in keys if k in m}
        print(json.dumps(out["sweeps"]["edge_chunk_bits"][str(B)]), flush=True)
    json.dump(out, open(RESULTS_DIR / "twostage_sensitivity.json", "w"), indent=2)
    return out


def main() -> None:
    run()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["main", "sensitivity",
                                           "prior_ablation", "cond_ablation"],
                        default="main")
    args = parser.parse_args()
    {"main": run, "sensitivity": run_sensitivity,
     "prior_ablation": run_prior_ablation, "cond_ablation": run_cond_ablation}[args.mode]()
