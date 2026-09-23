import copy
import json
import statistics
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .batching import to_torch, build_batches, evaluate
from .data import load_tudataset
from .models import VQGIN, make_model

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "benchmark.yaml"
RESULTS_DIR = ROOT / "results"

METHODS = ["GIN", "GCN", "SAGE", "GAT", "DGCNN", "DiffPool", "MLP", "Ours-VQ", "Continuous"]
# Auxiliary-loss multiplier per method (DiffPool's link-prediction + entropy
# losses follow the original paper's weight 1.0; the VQ commitment term keeps
# the config's commitment_beta).
AUX_WEIGHT = {"DiffPool": 1.0}


def train_eval(tg, model_factory, seed, cfg, device, return_model=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = len(tg)
    idx = np.random.permutation(n)
    ntr = int(n * cfg["training"]["train_ratio"])
    nva = int(n * cfg["training"]["val_ratio"])
    bs = cfg["training"]["batch_size"]
    eb = cfg["training"]["eval_batch_size"]
    tr_bt = build_batches(tg, idx[:ntr], bs, device)
    va_bt = build_batches(tg, idx[ntr:ntr + nva], eb, device)
    te_bt = build_batches(tg, idx[ntr + nva:], eb, device)

    model = model_factory().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    best_va, best_te = -1.0, 0.0
    for _ in range(cfg["training"]["epochs"]):
        model.train()
        order = np.random.permutation(len(tr_bt))
        for i in order:
            x, a, b, y = tr_bt[i]
            opt.zero_grad()
            logits, aux = model(x, a, b)
            loss = F.cross_entropy(logits, y) + cfg["training"]["commitment_beta"] * aux
            loss.backward()
            opt.step()
        va_acc = evaluate(model, va_bt)
        if va_acc >= best_va:
            best_va = va_acc
            best_te = evaluate(model, te_bt)
    return (best_te, model) if return_model else best_te


def collect_code_usage(model, tg, cfg, device):
    """E7: codebook utilization and assignment entropy over the whole dataset.

    Returns ``(utilization, entropy)`` where utilization is the fraction of the
    ``K`` codebook entries that receive at least one assignment.
    """
    model.eval()
    eb = cfg["training"]["eval_batch_size"]
    bt = build_batches(tg, list(range(len(tg))), eb, device)
    all_idx = []
    with torch.no_grad():
        for x, a, b, _ in bt:
            h = x
            for conv in model.convs:
                h = conv(h, a)
            _, idx, _ = model.quantizer(h)
            all_idx.append(idx)
    idx = torch.cat(all_idx)
    K = model.quantizer.codebook.size(0)
    uniq = idx.unique().numel()
    freq = torch.bincount(idx, minlength=K).float()
    freq = freq / max(freq.sum().item(), 1.0)
    ent = float(-(freq[freq > 0] * freq[freq > 0].log()).sum())
    return uniq / K, ent


def run() -> dict:
    cfg = yaml.safe_load(open(CONFIG_PATH, encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mc = cfg["model"]
    results = {}

    for ds in cfg["data"]["datasets"]:
        graphs, num_classes, in_dim = load_tudataset(ds)
        tg = to_torch(graphs, device)
        results[ds] = {}
        for method in METHODS:
            c = copy.deepcopy(cfg)
            if method in AUX_WEIGHT:
                c["training"]["commitment_beta"] = AUX_WEIGHT[method]
            accs = [
                train_eval(tg, (lambda m=method: make_model(m, in_dim, mc, num_classes)),
                           seed, c, device)
                for seed in cfg["training"]["seeds"]
            ]
            results[ds][method] = {
                "mean": statistics.mean(accs),
                "std": statistics.stdev(accs) if len(accs) > 1 else 0.0,
                "per_seed": [round(a, 6) for a in accs],  # E12: paired t-tests
            }
            print(f"[{ds}] {method}: {results[ds][method]['mean']:.4f} ± {results[ds][method]['std']:.4f}", flush=True)
        json.dump(results, open(RESULTS_DIR / "benchmark_results.json", "w"), indent=2)

    ds = cfg["scaling"]["dataset"]
    graphs, num_classes, in_dim = load_tudataset(ds)
    tg = to_torch(graphs, device)
    sweep = {}
    for k in cfg["scaling"]["codebook_sizes"]:
        accs = []
        last_model = None
        for seed in cfg["training"]["seeds"]:
            acc, model = train_eval(
                tg, (lambda: VQGIN(in_dim, mc["hidden"], num_classes, k, mc.get("num_layers", 2))),
                seed, cfg, device, return_model=True)
            accs.append(acc)
            last_model = model
        entry = {"mean": statistics.mean(accs),
                 "std": statistics.stdev(accs) if len(accs) > 1 else 0.0,
                 "per_seed": [round(a, 6) for a in accs]}
        if last_model is not None:
            util, ent = collect_code_usage(last_model, tg, cfg, device)
            entry["codebook_utilization"] = round(util, 4)
            entry["codebook_entropy"] = round(ent, 4)
        sweep[str(k)] = entry
        print(f"[scaling {ds}] codebook={k}: {entry}", flush=True)

    json.dump({"results": results, "scaling": sweep}, open(RESULTS_DIR / "benchmark_results.json", "w"), indent=2)

    lines = [""]
    lines.append("## Main results (test accuracy ↑)")
    lines.append("| Dataset | " + " | ".join(METHODS) + " |")
    lines.append("| --- |" + " --- |" * len(METHODS))
    for d, mres in results.items():
        vals = [f"{mres[m]['mean']:.4f}±{mres[m]['std']:.4f}" for m in METHODS]
        lines.append(f"| {d} | " + " | ".join(vals) + " |")
    lines += ["", f"## Codebook-size scaling axis ({ds}, Ours-VQ)", "| codebook_size | Test accuracy |", "| --- | --- |"]
    for k, r in sweep.items():
        lines.append(f"| {k} | {r['mean']:.4f} ± {r['std']:.4f} |")
    report = "\n".join(lines)
    open(RESULTS_DIR / "benchmark_report.md", "w", encoding="utf-8").write(report + "\n")
    print("\n" + report, flush=True)
    return {"results": results, "scaling": sweep}


def main() -> None:
    run()


if __name__ == "__main__":
    run()
