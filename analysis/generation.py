"""BFS ordering + explicit edge modeling + sparse regularization + full metrics.

Improvements over the initial version:
  - BFS node ordering (edges cluster near the diagonal -> better window modeling).
  - Edge existence modeled explicitly with BCE (+ sparsity penalty) instead of MSE.
  - Full generation metrics: validity / novelty / uniqueness + degree distribution,
    clustering coefficient, triangle count, connectivity.
  - Multi-dataset: MUTAG / PROTEINS (TUDataset) and QM9 (SMILES -> RDKit graphs).
  - Multi-seed evaluation (E13) with per-seed dumps and mean/std aggregation.
  - ``edge_head`` config switch restoring the v1 head (MSE + 0.5 threshold) for
    the edge-head ablation (E6): ``edge_head: bce`` (default, v2) | ``mse`` (v1).
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

from .dataset import load_dataset
from .metrics import summarize_graphs
from .prior import make_prior, train_prior, sample_tokens

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "generation.yaml"
RESULTS_DIR = ROOT / "results"
SEEDS_DIR = RESULTS_DIR / "generation_seeds"


# ---------------------------------------------------------------- graph utils
def bfs_order(adj: np.ndarray):
    n = adj.shape[0]
    if n == 0:
        return []
    start = int(np.argmax(adj.sum(1)))
    order = [start]
    visited = {start}
    frontier = [start]
    while frontier:
        nxt = []
        for u in frontier:
            for v in range(n):
                if adj[u, v] and v not in visited:
                    visited.add(v)
                    order.append(v)
                    nxt.append(v)
        frontier = nxt
    for v in range(n):
        if v not in visited:
            order.append(v)
    return order


def dfs_order(adj: np.ndarray):
    """Iterative preorder DFS from the highest-degree node (reviewer Q3)."""
    n = adj.shape[0]
    if n == 0:
        return []
    start = int(np.argmax(adj.sum(1)))
    order, visited = [], {start}
    stack = [start]
    while stack:
        u = stack.pop()
        order.append(u)
        nbrs = [v for v in range(n) if adj[u, v] and v not in visited]
        for v in nbrs:
            visited.add(v)
            stack.append(v)
    for v in range(n):
        if v not in visited:
            order.append(v)
    return order


def deg_desc_order(adj: np.ndarray):
    """Global degree-descending order (stable: ties broken by node id) (reviewer Q3)."""
    n = adj.shape[0]
    if n == 0:
        return []
    return [int(i) for i in np.argsort(-adj.sum(1), kind="stable")]


def random_order(adj: np.ndarray):
    """Uniformly random node order (seeded via ``np.random.seed``) (reviewer Q3)."""
    return list(np.random.permutation(adj.shape[0]))


def bfs_random_order(adj: np.ndarray):
    """Random-root BFS with random same-level tie-breaking (reviewer 2.5).

    Each call returns a fresh random order (seeded via ``np.random.seed``).
    """
    n = adj.shape[0]
    if n == 0:
        return []
    start = int(np.random.randint(0, n))
    order = [start]
    visited = {start}
    frontier = [start]
    while frontier:
        nxt = []
        for u in frontier:
            nbrs = [v for v in range(n) if adj[u, v] and v not in visited]
            np.random.shuffle(nbrs)
            for v in nbrs:
                visited.add(v)
                order.append(v)
                nxt.append(v)
        frontier = nxt
    for v in range(n):
        if v not in visited:
            order.append(v)
    return order


ORDER_FUNCS = {
    "bfs": bfs_order,
    "bfs_random": bfs_random_order,
    "dfs": dfs_order,
    "deg_desc": deg_desc_order,
    "random": random_order,
}


def build_contexts(adj: np.ndarray, feat: np.ndarray, window: int,
                   order: str = "bfs") -> np.ndarray:
    """Serialize a graph into per-node contexts following the chosen node order."""
    n = adj.shape[0]
    order = ORDER_FUNCS[order](adj)
    ctx = []
    for pos, i in enumerate(order):
        start = max(0, pos - window)
        prev = order[start:pos]
        e = np.zeros(window, dtype=np.float32)
        for k, j in enumerate(prev):
            e[window - len(prev) + k] = adj[i, j]
        ctx.append(np.concatenate([feat[i], e]).astype(np.float32))
    return np.stack(ctx)


# ---------------------------------------------------------------- models
class Tokenizer(nn.Module):
    """VQ-VAE over node contexts (feature one-hot + edge mask to previous window)."""

    def __init__(self, in_dim: int, window: int, hidden: int, codebook_size: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.window = window
        ctx_dim = in_dim + window
        self.encoder = nn.Sequential(nn.Linear(ctx_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.quantizer = VectorQuantizer(codebook_size, hidden)
        self.decoder = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, ctx_dim))

    def forward(self, ctx):
        z = self.encoder(ctx)
        z_q, idx, vq = self.quantizer(z)
        out = self.decoder(z_q)
        return out, idx, vq


# ---------------------------------------------------------------- training
def train_tokenizer(tok, contexts, in_dim, epochs, lr, beta, sparsity_beta, device,
                    edge_head="bce", bs=64):
    opt = torch.optim.Adam(tok.parameters(), lr=lr)
    x = torch.from_numpy(np.concatenate(contexts, 0)).to(device)
    n = x.size(0)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            out, _, vq = tok(x[idx])
            feat_logits = out[:, :in_dim]
            edge_logits = out[:, in_dim:]
            feat_loss = F.mse_loss(feat_logits, x[idx][:, :in_dim])
            if in_dim < out.shape[1]:
                if edge_head == "mse":
                    # v1 head: MSE regression + fixed 0.5 threshold at decode time.
                    edge_loss = F.mse_loss(edge_logits, x[idx][:, in_dim:])
                else:
                    edge_loss = F.binary_cross_entropy_with_logits(edge_logits, x[idx][:, in_dim:])
                sparsity = torch.sigmoid(edge_logits).mean()
            else:  # ctx_mode="feat": attribute-only context
                edge_loss = torch.zeros((), device=out.device)
                sparsity = torch.zeros((), device=out.device)
            loss = feat_loss + edge_loss + sparsity_beta * sparsity + beta * vq
            loss.backward()
            opt.step()
            opt.zero_grad()
    return tok


def tokenize_graph(tok, ctx):
    x = torch.from_numpy(ctx).to(next(tok.parameters()).device)
    _, idx, _ = tok(x)
    return idx.tolist()


def decode_token(tok, token_id, in_dim, window, device, edge_head="bce"):
    z = tok.quantizer.codebook[token_id]
    out = tok.decoder(z)
    feat_logits = out[:in_dim]
    edge_logits = out[in_dim:]
    feat = torch.zeros(in_dim, device=device)
    feat[feat_logits.argmax()] = 1.0
    edge_prob = torch.sigmoid(edge_logits)
    if edge_head == "mse":
        # v1: fixed 0.5 threshold (biases toward the majority no-edge class).
        edge = (edge_prob > 0.5).float()
    else:
        # v2: Bernoulli-sample from the BCE-calibrated probability, which aligns
        # the generated edge density with the data prior.
        edge = (torch.rand_like(edge_prob) < edge_prob).float()
    return feat.cpu().numpy(), edge.cpu().numpy()


def generate(tok, prior, node_counts, num_samples, in_dim, window, device, sos, eos,
             max_nodes=128, temperature=1.0, edge_head="bce"):
    tok.eval(); prior.eval()
    generated = []
    counts = np.random.choice(node_counts, num_samples)
    with torch.no_grad():
        for cnt in counts:
            cnt = min(int(cnt), max_nodes)
            tokens = sample_tokens(prior, cnt, None, device, sos, eos, temperature=temperature)
            feats, edges = [], []
            for t in tokens:
                f, e = decode_token(tok, t, in_dim, window, device, edge_head=edge_head)
                feats.append(f); edges.append(e)
            generated.append((feats, edges))
    return generated


def build_graph(feats, edges, window):
    n = len(feats)
    if n == 0:
        return None
    in_dim = len(feats[0])
    feat = np.array(feats)
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        start = max(0, i - window)
        prev = list(range(start, i))
        for k, j in enumerate(prev):
            v = edges[i][window - len(prev) + k]
            adj[i, j] = adj[j, i] = v
    return adj, feat


# ---------------------------------------------------------------- metrics
SIGNATURE_BY_DATASET = {"QM9": "qm9", "MUTAG": "isotope"}  # molecular graphs -> RDKit SMILES


def evaluate(generated, train_graphs, signature="wl"):
    gs = []
    for feats, edges in generated:
        if not feats:
            continue
        g = build_graph(feats, edges, len(edges[0]))
        if g is None:
            continue
        gs.append(g)
    return summarize_graphs(gs, train_graphs, include_hist_and_conn=True, signature=signature)


# ---------------------------------------------------------------- runs
def run_one(dataset: str, cfg: dict, device: str, overrides: dict = None,
            collect_graphs: bool = False, graphs: list = None):
    """Train and evaluate the one-stage generator once (single seed, from cfg['seed']).

    ``graphs`` optionally overrides the dataset (e.g. the 80% train split used by
    the P1 same-split protocol). With ``collect_graphs=True`` the built (adj, feat)
    graph list is returned as a second element for structural-metric evaluation.
    """
    cfg = dict(cfg)
    if overrides:
        cfg.update(overrides)
    window = cfg["window"]
    edge_head = cfg.get("edge_head", "bce")
    order = cfg.get("order", "bfs")
    prior_type = cfg.get("prior_type", "gru")
    prior_hidden = cfg.get("prior_hidden", cfg["hidden"])
    if graphs is None:
        graphs = load_dataset(dataset, cfg.get("n_qm9", 5000))
    in_dim = graphs[0][1].shape[1]
    order_aug = cfg.get("order_aug", 1)
    ctx_mode = cfg.get("ctx_mode", "full")
    if ctx_mode == "feat":  # ablation: attribute-only quantization, no local structure
        contexts = [x.astype(np.float32) for _, x, _ in graphs]
        eff_window = 0
    else:
        contexts = []
        for a, x, _ in graphs:
            for _ in range(order_aug):
                contexts.append(build_contexts(a, x, window, order=order))
        eff_window = window
    node_counts = [a.shape[0] for a, _, _ in graphs]

    tok = Tokenizer(in_dim, eff_window, cfg["hidden"], cfg["codebook_size"]).to(device)
    tok = train_tokenizer(tok, contexts, in_dim, cfg["stage1_epochs"], cfg["lr"],
                          cfg["commitment_beta"], cfg["sparsity_beta"], device, edge_head=edge_head)

    vocab = cfg["codebook_size"] + 2
    sos, eos = cfg["codebook_size"], cfg["codebook_size"] + 1
    all_tokens = [tokenize_graph(tok, c) for c in contexts]
    prior = make_prior(vocab, prior_hidden, 0, prior_type).to(device)
    prior = train_prior(prior, all_tokens, cfg["stage2_epochs"], cfg["lr"], device, sos, eos)

    g = generate(tok, prior, node_counts, cfg["num_samples"], in_dim, eff_window, device,
                 sos, eos, max_nodes=cfg.get("max_nodes", 128),
                 temperature=cfg.get("temperature", 1.0), edge_head=edge_head)
    signature = SIGNATURE_BY_DATASET.get(dataset, "wl")
    if cfg.get("dump_samples"):
        # E10: keep the first N generated graphs for the visual gallery
        import os
        dump_dir = Path(os.environ.get("DUMP_DIR", RESULTS_DIR)) / "samples"
        dump_dir.mkdir(parents=True, exist_ok=True)
        out = []
        for feats, edges in g[:cfg.get("dump_samples_n", 20)]:
            if not feats:
                continue
            gr = build_graph(feats, edges, len(edges[0]))
            if gr is not None:
                out.append(gr)
        np.savez_compressed(dump_dir / f"{dataset}_onestage.npz",
                            **{f"g{k}": np.asarray(gr[0]) for k, gr in enumerate(out)})
    metrics = evaluate(g, graphs, signature=signature)
    if collect_graphs:
        gs = []
        for feats, edges in g:
            if not feats:
                continue
            gr = build_graph(feats, edges, len(edges[0]))
            if gr is not None:
                gs.append(gr)
        return metrics, gs
    return metrics


# ---------------------------------------------------------------- order sensitivity (P1, reviewer 2.5)
def sequence_nll(prior, tokens, sos, eos, device):
    """NLL in nats/token of a token sequence under the AR prior (incl. EOS)."""
    prior.eval()
    seq = torch.tensor([sos] + list(tokens) + [eos], dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = prior(seq, None)[:, :-1].reshape(-1, prior.vocab)
        target = seq[:, 1:].reshape(-1)
        nll = F.cross_entropy(logits, target, reduction="sum").item()
    return nll / max(len(tokens) + 1, 1)


def heldout_order_nll(dataset, cfg, device, k_orders=8, seeds=(0,)):
    """Multi-order held-out sequence NLL (reviewer 2.5): per test graph, the AR
    prior scores ``k_orders`` fresh random-BFS serializations; report the mean
    NLL and the log-mean-exp aggregate over orders."""
    from .dataset import split_indices
    graphs = load_dataset(dataset, cfg.get("n_qm9", 5000))
    window = cfg["window"]
    edge_head = cfg.get("edge_head", "bce")
    order = cfg.get("order", "bfs")
    out = {"dataset": dataset, "order": order, "k_orders": k_orders, "seeds": {}}
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        tr_idx, va_idx, te_idx = split_indices(len(graphs), seed)
        train = [graphs[i] for i in tr_idx]
        order_aug = cfg.get("order_aug", 1)
        contexts = []
        for a, x, _ in train:
            for _ in range(order_aug):
                contexts.append(build_contexts(a, x, window, order=order))
        in_dim = graphs[0][1].shape[1]
        tok = Tokenizer(in_dim, window, cfg["hidden"], cfg["codebook_size"]).to(device)
        tok = train_tokenizer(tok, contexts, in_dim, cfg["stage1_epochs"], cfg["lr"],
                              cfg["commitment_beta"], cfg["sparsity_beta"], device,
                              edge_head=edge_head)
        vocab = cfg["codebook_size"] + 2
        sos, eos = cfg["codebook_size"], cfg["codebook_size"] + 1
        all_tokens = [tokenize_graph(tok, c) for c in contexts]
        prior = make_prior(vocab, cfg.get("prior_hidden", cfg["hidden"]), 0,
                           cfg.get("prior_type", "gru")).to(device)
        prior = train_prior(prior, all_tokens, cfg["stage2_epochs"], cfg["lr"], device, sos, eos)
        nlls_mean, nlls_lme = [], []
        for i in list(va_idx) + list(te_idx):
            a, x, _ = graphs[i]
            per = []
            for _ in range(k_orders):
                ctx = build_contexts(a, x, window, order=order)
                per.append(sequence_nll(prior, tokenize_graph(tok, ctx), sos, eos, device))
            nlls_mean.append(float(np.mean(per)))
            nlls_lme.append(-float(np.log(np.mean(np.exp(-np.asarray(per))))))
        out["seeds"][str(seed)] = {
            "mean_nll": round(float(np.mean(nlls_mean)), 4),
            "log_mean_exp_nll": round(float(np.mean(nlls_lme)), 4),
            "n_graphs": len(nlls_mean)}
        print(f"[order_nll/{dataset}/seed{seed}]", json.dumps(out["seeds"][str(seed)]), flush=True)
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
    window = cfg["window"]
    SEEDS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for ds in cfg["datasets"]:
        print(f"\n=== {ds} ===", flush=True)
        graphs = load_dataset(ds, cfg.get("n_qm9", 5000))
        in_dim = graphs[0][1].shape[1]
        contexts = [build_contexts(a, x, window) for a, x, _ in graphs]
        node_counts = [a.shape[0] for a, _, _ in graphs]

        per_seed = {}
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            print(f"  seed {seed}...", flush=True)
            m = run_one(ds, dict(cfg, seed=seed), device)
            per_seed[str(seed)] = m
            json.dump(m, open(SEEDS_DIR / f"{ds}_{seed}.json", "w"), indent=2)
            print(json.dumps(m, indent=2), flush=True)
        agg = _aggregate(per_seed)
        all_results[ds] = agg
        print(f"[{ds}] aggregated:", json.dumps(agg["metrics"], indent=2), flush=True)

    json.dump(all_results, open(RESULTS_DIR / "generation_results.json", "w"), indent=2)
    return all_results


def run_ablation() -> dict:
    """E6: edge-head calibration ablation on MUTAG (v1 MSE+thr / v2 BCE+Bernoulli / v2+sparsity)."""
    cfg = yaml.safe_load(open(CONFIG_PATH, encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = cfg.get("seeds", [0, 1, 2])
    dataset = "MUTAG"
    variants = [
        ("v1_mse_threshold", {"edge_head": "mse", "sparsity_beta": 0.0}),
        ("v2_bce_bernoulli", {"edge_head": "bce", "sparsity_beta": 0.0}),
        ("v2_bce_sparsity03", {"edge_head": "bce", "sparsity_beta": 0.3}),
    ]
    out = {"dataset": dataset, "variants": {}}
    for name, overrides in variants:
        print(f"\n=== {name} ===", flush=True)
        per_seed = {}
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            m = run_one(dataset, dict(cfg, seed=seed), device, overrides)
            per_seed[str(seed)] = m
            json.dump(m, open(SEEDS_DIR / f"ablation_{name}_{seed}.json", "w"), indent=2)
        out["variants"][name] = _aggregate(per_seed)
        print(f"[{name}]:", json.dumps(out["variants"][name]["metrics"]), flush=True)
        json.dump(out, open(RESULTS_DIR / "edge_head_ablation.json", "w"), indent=2)
    return out


def run_prior_ablation() -> dict:
    """Reviewer Q1 (one-stage): GRU-32 baseline (existing generation_seeds) vs.
    2-layer transformer priors at hidden 32 and 128, three seeds per variant."""
    cfg = yaml.safe_load(open(CONFIG_PATH, encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = cfg.get("seeds", [0, 1, 2])
    variants = [
        ("tf32", {"prior_type": "transformer", "prior_hidden": 32}),
        ("tf128", {"prior_type": "transformer", "prior_hidden": 128}),
    ]
    SEEDS_DIR.mkdir(parents=True, exist_ok=True)
    out = {"baseline": "gru32 (generation_seeds)", "variants": {}}
    for name, ov in variants:
        out["variants"][name] = {}
        for ds in ["MUTAG", "PROTEINS", "QM9"]:
            print(f"\n=== prior {name} / {ds} ===", flush=True)
            per_seed = {}
            for seed in seeds:
                torch.manual_seed(seed); np.random.seed(seed)
                m = run_one(ds, dict(cfg, seed=seed, **ov), device)
                per_seed[str(seed)] = m
                json.dump(m, open(SEEDS_DIR / f"prior_{name}_{ds}_{seed}.json", "w"), indent=2)
                print(f"[{name}/{ds}/seed{seed}]", json.dumps(m), flush=True)
            out["variants"][name][ds] = _aggregate(per_seed)
            json.dump(out, open(RESULTS_DIR / "prior_ablation.json", "w"), indent=2)
    return out


def run_order_ablation() -> dict:
    """Reviewer Q3: serialization-order sensitivity of one-stage generation on MUTAG
    (BFS baseline = existing generation_seeds; dfs/deg_desc/random rerun here)."""
    cfg = yaml.safe_load(open(CONFIG_PATH, encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = cfg.get("seeds", [0, 1, 2])
    dataset = "MUTAG"
    orders = ["dfs", "deg_desc", "random"]
    SEEDS_DIR.mkdir(parents=True, exist_ok=True)
    out = {"dataset": dataset, "baseline": "bfs (generation_seeds)", "orders": {}}
    for order in orders:
        print(f"\n=== order {order} / {dataset} ===", flush=True)
        per_seed = {}
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            m = run_one(dataset, dict(cfg, seed=seed, order=order), device)
            per_seed[str(seed)] = m
            json.dump(m, open(SEEDS_DIR / f"order_{order}_{dataset}_{seed}.json", "w"), indent=2)
            print(f"[{order}/seed{seed}]", json.dumps(m), flush=True)
        out["orders"][order] = _aggregate(per_seed)
        json.dump(out, open(RESULTS_DIR / "order_ablation.json", "w"), indent=2)
    return out


def run_sensitivity() -> dict:
    """E9: single-factor sensitivity sweeps on MUTAG (window / commitment beta / temperature)."""
    cfg = yaml.safe_load(open(CONFIG_PATH, encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = "MUTAG"
    sweeps = [
        ("window", [4, 8, 16]),
        ("commitment_beta", [0.1, 0.25, 1.0]),
        ("temperature", [0.8, 1.0, 1.2]),
    ]
    keys = ["validity", "mean_degree", "triangle", "connectivity"]
    out = {"dataset": dataset, "sweeps": {}}
    for param, values in sweeps:
        print(f"\n=== sweep {param} ===", flush=True)
        row = {}
        for v in values:
            torch.manual_seed(0); np.random.seed(0)
            m = run_one(dataset, dict(cfg, seed=0, **{param: v}), device)
            row[str(v)] = {k: m[k] for k in keys if k in m}
            print(f"  {param}={v}:", json.dumps(row[str(v)]), flush=True)
        out["sweeps"][param] = row
    json.dump(out, open(RESULTS_DIR / "sensitivity.json", "w"), indent=2)
    return out


def main() -> None:
    run()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["main", "ablation", "sensitivity",
                                           "prior_ablation", "order_ablation"],
                        default="main")
    args = parser.parse_args()
    {"main": run, "ablation": run_ablation, "sensitivity": run_sensitivity,
     "prior_ablation": run_prior_ablation, "order_ablation": run_order_ablation}[args.mode]()
