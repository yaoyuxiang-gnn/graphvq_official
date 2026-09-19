import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from model.vector_quantizer import VectorQuantizer

from .batching import to_torch, build_batches, evaluate
from .data import load_tudataset
from .layers import GINLayer, mean_pool

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "transfer.yaml"
RESULTS_DIR = ROOT / "results"


class TransferModel(nn.Module):
    """Feature adapter -> GIN encoder -> [VQ codebook] -> classifier."""

    def __init__(self, in_dim: int, hidden: int, num_classes: int, use_vq: bool, codebook_size: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.use_vq = use_vq
        self.codebook_size = codebook_size
        self.adapter = nn.Linear(in_dim, hidden)
        self.gin1 = GINLayer(hidden, hidden)
        self.gin2 = GINLayer(hidden, hidden)
        self.quantizer = VectorQuantizer(codebook_size, hidden) if use_vq else None
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x, adj, batch):
        h = self.adapter(x)
        h = self.gin1(h, adj)
        h = self.gin2(h, adj)
        aux = torch.zeros((), device=x.device)
        if self.quantizer is not None:
            h, _, aux = self.quantizer(h)
        g = mean_pool(h, batch)
        return self.classifier(g), aux

    def shared_params(self):
        ps = list(self.gin1.parameters()) + list(self.gin2.parameters())
        if self.quantizer is not None:
            ps += list(self.quantizer.parameters())
        return ps

    def head_params(self):
        return list(self.adapter.parameters()) + list(self.classifier.parameters())


def train(model, batches, opt, epochs, beta):
    for _ in range(epochs):
        model.train()
        order = np.random.permutation(len(batches))
        for i in order:
            x, a, b, y = batches[i]
            opt.zero_grad()
            logits, aux = model(x, a, b)
            loss = F.cross_entropy(logits, y) + beta * aux
            loss.backward()
            opt.step()


def sample_kshot(labels, K, seed):
    rng = np.random.RandomState(seed)
    by_class = defaultdict(list)
    for i, y in enumerate(labels):
        by_class[y].append(i)
    sel = []
    for c, idxs in by_class.items():
        sel.extend(rng.choice(idxs, min(K, len(idxs)), replace=False).tolist())
    return sorted(sel)


def pretrain(in_dim, num_classes, use_vq, cfg, tg_src, tr_idx, device):
    model = TransferModel(in_dim, cfg["hidden"], num_classes, use_vq, cfg["codebook_size"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    tr_bt = build_batches(tg_src, tr_idx, cfg["batch_size"], device)
    train(model, tr_bt, opt, cfg["pretrain_epochs"], cfg["commitment_beta"])
    return model


def transfer(pretrained, in_dim_t, num_classes_t, tg_tgt, kshot_idx, te_idx, cfg, device,
             freeze_codebook=True, freeze_encoder=True):
    """Copy the shared part, re-init the head for the target, freeze shared, train head.

    ``freeze_codebook=False`` unfreezes the VQ codebook (E8 variant); ``freeze_encoder=False``
    additionally unfreezes the GIN encoder (tune-all variant).
    """
    m = TransferModel(in_dim_t, pretrained.hidden, num_classes_t, pretrained.use_vq, pretrained.codebook_size).to(device)
    m.gin1.load_state_dict(pretrained.gin1.state_dict())
    m.gin2.load_state_dict(pretrained.gin2.state_dict())
    if m.quantizer is not None:
        m.quantizer.load_state_dict(pretrained.quantizer.state_dict())
    for p in m.shared_params():
        p.requires_grad = False
    if m.quantizer is not None and not freeze_codebook:
        for p in m.quantizer.parameters():
            p.requires_grad = True
    if not freeze_encoder:
        for p in list(m.gin1.parameters()) + list(m.gin2.parameters()):
            p.requires_grad = True
    if not freeze_codebook or not freeze_encoder:
        trainable = [p for p in m.parameters() if p.requires_grad]
        opt = torch.optim.Adam(trainable, lr=cfg["lr"])
    else:
        opt = torch.optim.Adam(m.head_params(), lr=cfg["lr"])
    tr_bt = build_batches(tg_tgt, kshot_idx, cfg["batch_size"], device)
    te_bt = build_batches(tg_tgt, te_idx, cfg["eval_batch_size"], device)
    train(m, tr_bt, opt, cfg["fewshot_epochs"], cfg["commitment_beta"])
    return evaluate(m, te_bt)


def from_scratch(in_dim_t, num_classes_t, tg_tgt, kshot_idx, te_idx, cfg, device):
    m = TransferModel(in_dim_t, cfg["hidden"], num_classes_t, False, cfg["codebook_size"]).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"])
    tr_bt = build_batches(tg_tgt, kshot_idx, cfg["batch_size"], device)
    te_bt = build_batches(tg_tgt, te_idx, cfg["eval_batch_size"], device)
    train(m, tr_bt, opt, cfg["fewshot_epochs"], cfg["commitment_beta"])
    return evaluate(m, te_bt)


def run() -> dict:
    cfg = yaml.safe_load(open(CONFIG_PATH, encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = {}

    def get(ds):
        if ds not in cache:
            graphs, nc, ind = load_tudataset(ds)
            cache[ds] = (to_torch(graphs, device), nc, ind)
        return cache[ds]

    results = {}
    for src, tgt in cfg["pairs"]:
        tg_s, nc_s, ind_s = get(src)
        tg_t, nc_t, ind_t = get(tgt)
        n_s, n_t = len(tg_s), len(tg_t)
        results[f"{src}->{tgt}"] = {}
        for seed in cfg["seeds"]:
            torch.manual_seed(seed)
            np.random.seed(seed)
            s_idx = np.random.permutation(n_s)
            t_idx = np.random.permutation(n_t)
            ntr_s = int(n_s * cfg["source_train_ratio"])
            # The target pool uses two equal 10% halves only to keep the exact
            # integer-truncation behaviour of the original int(n*0.1) + int(n*0.1).
            half = cfg["target_pool_ratio"] / 2.0
            pool_len = 2 * int(n_t * half)
            tr_s = s_idx[:ntr_s]
            t_pool = t_idx[:pool_len]
            te_idx = t_idx[pool_len:]
            labels_t = [tg_t[i][2] for i in t_pool]

            for method in ["Ours-VQ", "GIN"]:
                use_vq = (method == "Ours-VQ")
                key = f"{method}"
                results[f"{src}->{tgt}"].setdefault(key, {})
                pretrained = pretrain(ind_s, nc_s, use_vq, cfg, tg_s, tr_s, device)
                for K in cfg["K_values"]:
                    ks_labels = sample_kshot(labels_t, K, seed)   # positions in t_pool
                    ks = [t_pool[i] for i in ks_labels]            # map to tg_t indices
                    acc = transfer(pretrained, ind_t, nc_t, tg_t, ks, te_idx, cfg, device)
                    results[f"{src}->{tgt}"][key].setdefault(f"K={K}", []).append(acc)
                # E8 variants (only for the VQ model; reuse the same pretrained weights)
                if use_vq:
                    for vname, fcb, fe in [("Ours-VQ-tune-cb", False, True),
                                           ("Ours-VQ-tune-all", False, False)]:
                        results[f"{src}->{tgt}"].setdefault(vname, {})
                        for K in cfg["K_values"]:
                            ks_labels = sample_kshot(labels_t, K, seed)
                            ks = [t_pool[i] for i in ks_labels]
                            acc = transfer(pretrained, ind_t, nc_t, tg_t, ks, te_idx, cfg, device,
                                           freeze_codebook=fcb, freeze_encoder=fe)
                            results[f"{src}->{tgt}"][vname].setdefault(f"K={K}", []).append(acc)
            # from-scratch (no pretrain)
            for K in cfg["K_values"]:
                ks_labels = sample_kshot(labels_t, K, seed)
                ks = [t_pool[i] for i in ks_labels]
                acc = from_scratch(ind_t, nc_t, tg_t, ks, te_idx, cfg, device)
                results[f"{src}->{tgt}"].setdefault("FromScratch", {}).setdefault(f"K={K}", []).append(acc)

    # aggregate mean±std
    agg = {}
    for pair, methods in results.items():
        agg[pair] = {}
        for method, ks in methods.items():
            agg[pair][method] = {k: {"mean": statistics.mean(v), "std": statistics.stdev(v) if len(v) > 1 else 0.0,
                                     "per_seed": [round(a, 6) for a in v]} for k, v in ks.items()}
    json.dump(agg, open(RESULTS_DIR / "transfer_results.json", "w"), indent=2)
    json.dump(results, open(RESULTS_DIR / "transfer_per_seed.json", "w"), indent=2)

    for pair, methods in agg.items():
        print(f"\n=== {pair} ===", flush=True)
        for method, ks in methods.items():
            for k, r in ks.items():
                print(f"  {method} {k}: {r['mean']:.4f} ± {r['std']:.4f}", flush=True)
    return agg


def main() -> None:
    run()


if __name__ == "__main__":
    run()
