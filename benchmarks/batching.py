"""Shared batching and evaluation helpers for graph-classification experiments."""
import torch


def to_torch(graphs, device):
    """Convert numpy graphs to torch tensors once."""
    return [(torch.from_numpy(a).to(device), torch.from_numpy(x).to(device), y)
            for (a, x, y) in graphs]


def build_batches(tg, indices, bs, device):
    """Pre-build mini-batch tensors for a fixed index split."""
    batches = []
    for s in range(0, len(indices), bs):
        sub = [tg[i] for i in indices[s:s + bs]]
        a = torch.block_diag(*[g[0] for g in sub])
        x = torch.cat([g[1] for g in sub], 0)
        b = torch.cat([torch.full((g[0].shape[0],), k, dtype=torch.long, device=device)
                       for k, g in enumerate(sub)], 0)
        y = torch.tensor([g[2] for g in sub], dtype=torch.long, device=device)
        batches.append((x, a, b, y))
    return batches


def evaluate(model, batches):
    """Return accuracy over pre-built batches."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for (x, a, b, y) in batches:
            logits, _ = model(x, a, b)
            correct += int((logits.argmax(1) == y).sum().item())
            total += y.size(0)
    return correct / total
