import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.vector_quantizer import VectorQuantizer

from .layers import GINLayer, GCNLayer, SAGELayer, GATLayer, mean_pool, sum_pool, code_histogram


class GNNClassifier(nn.Module):
    """Shared backbone: stacked conv layers -> [VQ] -> mean readout -> linear classifier."""

    def __init__(self, in_dim: int, hidden: int, num_classes: int, num_layers: int,
                 conv_cls, use_vq: bool = False, codebook_size: int = None) -> None:
        super().__init__()
        self.convs = nn.ModuleList([conv_cls(in_dim if i == 0 else hidden, hidden) for i in range(num_layers)])
        self.quantizer = VectorQuantizer(codebook_size, hidden) if use_vq else None
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x, adj, batch):
        h = x
        for conv in self.convs:
            h = conv(h, adj)
        aux = torch.zeros((), device=x.device)
        if self.quantizer is not None:
            h, _, aux = self.quantizer(h)
        g = mean_pool(h, batch)
        return self.classifier(g), aux


class GIN(GNNClassifier):
    """Baseline: 2-layer GIN + mean readout + linear classifier."""

    def __init__(self, in_dim: int, hidden: int, num_classes: int, num_layers: int = 2) -> None:
        super().__init__(in_dim, hidden, num_classes, num_layers, GINLayer)


class GCN(GNNClassifier):
    """Baseline: 2-layer GCN + mean readout + linear classifier."""

    def __init__(self, in_dim: int, hidden: int, num_classes: int, num_layers: int = 2) -> None:
        super().__init__(in_dim, hidden, num_classes, num_layers, GCNLayer)


class SAGE(GNNClassifier):
    """Baseline: 2-layer GraphSAGE + mean readout + linear classifier."""

    def __init__(self, in_dim: int, hidden: int, num_classes: int, num_layers: int = 2) -> None:
        super().__init__(in_dim, hidden, num_classes, num_layers, SAGELayer)


class VQGIN(GNNClassifier):
    def __init__(self, in_dim: int, hidden: int, num_classes: int, codebook_size: int, num_layers: int = 2) -> None:
        super().__init__(in_dim, hidden, num_classes, num_layers, GINLayer, use_vq=True, codebook_size=codebook_size)


class VQGINSum(nn.Module):
    """GraphVQ-D V1: GIN encoder -> VQ tokenization -> **sum** readout -> classifier.

    Swaps the mean readout for sum (GIN's counting-preserving readout) and is
    depth/configurable; otherwise identical to ``VQGIN``.
    """

    def __init__(self, in_dim: int, hidden: int, num_classes: int, codebook_size: int, num_layers: int = 2) -> None:
        super().__init__()
        self.convs = nn.ModuleList([GINLayer(in_dim if i == 0 else hidden, hidden) for i in range(num_layers)])
        self.quantizer = VectorQuantizer(codebook_size, hidden)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x, adj, batch):
        h = x
        for conv in self.convs:
            h = conv(h, adj)
        h, _, aux = self.quantizer(h)
        g = sum_pool(h, batch)
        return self.classifier(g), aux


class GINVQDual(nn.Module):
    """GraphVQ-D V2: multi-round VQ refinement + dual readout (the neural WL kernel).

    ``num_layers`` GIN rounds refine the node features; after every round the
    features are quantized by a round-specific codebook (learned ``colors``) and
    the discrete tokens propagate to the next round, mirroring WL color
    refinement. The readout concatenates (i) the per-round L2-normalized
    code-index histograms -- exact multiset counting, WL's core operation --
    and (ii) the sum-pooled quantized embeddings, which carry gradients back to
    the encoder (the argmin of the histograms is not differentiable by itself).
    """

    def __init__(self, in_dim: int, hidden: int, num_classes: int, num_layers: int = 3,
                 codebook_size: int = 64) -> None:
        super().__init__()
        self.convs = nn.ModuleList([GINLayer(in_dim if i == 0 else hidden, hidden) for i in range(num_layers)])
        self.quantizers = nn.ModuleList([VectorQuantizer(codebook_size, hidden) for _ in range(num_layers)])
        self.k = codebook_size
        self.classifier = nn.Sequential(
            nn.Linear(num_layers * codebook_size + hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x, adj, batch):
        h = x
        aux = torch.zeros((), device=x.device)
        hists = []
        for i, conv in enumerate(self.convs):
            h = conv(h, adj)
            h, idx, vq_loss = self.quantizers[i](h)
            hists.append(code_histogram(idx, batch, self.k))
            aux = aux + vq_loss / len(self.convs)  # mean per round: keep total commitment comparable to V1
        emb = sum_pool(h, batch)
        feat = torch.cat(hists + [emb], dim=1)
        return self.classifier(feat), aux


def _soft_counts(soft: torch.Tensor, batch: torch.Tensor, k: int) -> torch.Tensor:
    """Per-graph L2-normalized soft-assignment histograms, fully differentiable."""
    b = int(batch.max().item()) + 1
    idx = batch[:, None] * k + torch.arange(k, device=soft.device)[None, :]
    flat = torch.zeros(b * k, device=soft.device)
    flat.index_add_(0, idx.reshape(-1), soft.reshape(-1))
    cnt = flat.view(b, k)
    return cnt / cnt.norm(dim=1, keepdim=True).clamp(min=1e-8)


class GINVQSoftDual(nn.Module):
    """GraphVQ-D V2-soft: multi-round refinement + differentiable dual readout.

    Same as ``GINVQDual`` except the counting arm is built from soft
    assignments ``softmax(-dist/tau)`` instead of hard argmax, so the encoder
    receives gradients through the histograms themselves (no straight-through
    dependence). Hard quantization is retained for the embedding arm only.
    """

    def __init__(self, in_dim: int, hidden: int, num_classes: int, num_layers: int = 3,
                 codebook_size: int = 64, tau: float = 0.5) -> None:
        super().__init__()
        self.convs = nn.ModuleList([GINLayer(in_dim if i == 0 else hidden, hidden) for i in range(num_layers)])
        self.codebooks = nn.Parameter(torch.randn(num_layers, codebook_size, hidden) * 0.1)
        self.k = codebook_size
        self.tau = tau
        self.classifier = nn.Sequential(
            nn.Linear(num_layers * codebook_size + hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x, adj, batch):
        h = x
        hists = []
        for i, conv in enumerate(self.convs):
            h = conv(h, adj)
            cb = self.codebooks[i]
            dist = (h ** 2).sum(-1, keepdim=True) + (cb ** 2).sum(-1).unsqueeze(0) - 2.0 * h @ cb.t()
            soft = torch.softmax(-dist / self.tau, dim=-1)
            hists.append(_soft_counts(soft, batch, self.k))
            idx = dist.argmin(dim=-1)  # hard assignment for the embedding arm (straight-through)
            zq = cb[idx] + (h - cb[idx]).detach()
            h = zq
        emb = sum_pool(h, batch)
        feat = torch.cat(hists + [emb], dim=1)
        return self.classifier(feat), torch.zeros((), device=x.device)


class GAT(GNNClassifier):
    """Baseline: 2-layer GAT (single head) + mean readout + linear classifier."""

    def __init__(self, in_dim: int, hidden: int, num_classes: int, num_layers: int = 2) -> None:
        super().__init__(in_dim, hidden, num_classes, num_layers, GATLayer)


class GINSum(nn.Module):
    """Baseline: GIN encoder + **sum** readout (GIN's counting-preserving readout)."""

    def __init__(self, in_dim: int, hidden: int, num_classes: int, num_layers: int = 2) -> None:
        super().__init__()
        self.convs = nn.ModuleList([GINLayer(in_dim if i == 0 else hidden, hidden) for i in range(num_layers)])
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x, adj, batch):
        h = x
        for conv in self.convs:
            h = conv(h, adj)
        g = sum_pool(h, batch)
        return self.classifier(g), torch.zeros((), device=x.device)


class MLP(nn.Module):
    """Baseline: structure-agnostic node-feature MLP (bag-of-nodes sanity check).

    Mean-pools the raw node features without any message passing, so it
    quantifies how much of the benchmark is solvable from node features alone.
    """

    def __init__(self, in_dim: int, hidden: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, num_classes))

    def forward(self, x, adj, batch):
        g = mean_pool(x, batch)
        return self.net(g), torch.zeros((), device=x.device)


class DGCNN(nn.Module):
    """Baseline: DGCNN (Zhang et al., AAAI 2018) -- conv layers + SortPool + 1D CNN.

    Nodes are sorted by descending last-layer feature magnitude within each
    graph, truncated/padded to ``k``, and a two-layer 1D CNN with max pooling
    produces the graph embedding.
    """

    def __init__(self, in_dim: int, hidden: int, num_classes: int, num_layers: int = 2, k: int = 30) -> None:
        super().__init__()
        self.convs = nn.ModuleList([GCNLayer(in_dim if i == 0 else hidden, hidden) for i in range(num_layers)])
        self.k = k
        self.conv1d1 = nn.Conv1d(hidden, 32, kernel_size=5, padding=2)
        self.conv1d2 = nn.Conv1d(32, 32, kernel_size=5, padding=2)
        self.classifier = nn.Linear(32, num_classes)

    def sort_pool(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """SortPool: per graph, keep the top-k nodes by descending feature sum."""
        n_g = int(batch.max().item()) + 1
        out = torch.zeros(n_g, self.k, h.size(1), device=h.device)
        for g in range(n_g):
            hg = h[batch == g]
            order = torch.argsort(hg.sum(dim=1), descending=True)[: self.k]
            out[g, : order.size(0)] = hg[order]
        return out

    def forward(self, x, adj, batch):
        h = x
        for conv in self.convs:
            h = conv(h, adj)
        s = self.sort_pool(h, batch).transpose(1, 2)  # (B, D, k)
        z = F.relu(self.conv1d1(s))
        z = F.relu(self.conv1d2(z))
        z = z.max(dim=2).values
        return self.classifier(z), torch.zeros((), device=x.device)


class DiffPool(nn.Module):
    """Baseline: DiffPool (Ying et al., NeurIPS 2018) -- differentiable clustering.

    Two GCN layers embed the nodes, a GCN-style assignment network produces a
    soft cluster assignment (25% of nodes per graph, capped), the graph is
    coarsened, one post-pool GCN layer refines the cluster features, and a mean
    readout feeds the classifier. The auxiliary link-prediction (Frobenius) and
    entropy losses follow the original paper (weights 1.0 / 0.5).
    """

    def __init__(self, in_dim: int, hidden: int, num_classes: int, num_layers: int = 2,
                 pool_ratio: float = 0.25, max_clusters: int = 25) -> None:
        super().__init__()
        self.convs = nn.ModuleList([GCNLayer(in_dim if i == 0 else hidden, hidden) for i in range(num_layers)])
        self.pool_ratio = pool_ratio
        self.max_clusters = max_clusters
        self.embed_pool = nn.Linear(hidden, hidden)
        self.assign_pool = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, max_clusters))
        self.conv_after = GCNLayer(hidden, hidden)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x, adj, batch):
        h = x
        for conv in self.convs:
            h = conv(h, adj)
        n_g = int(batch.max().item()) + 1

        # Per-graph soft assignments, padded to the largest cluster count of the batch.
        sizes = []
        s_list = []
        lp = 0.0
        ent = 0.0
        for g in range(n_g):
            m = batch == g
            hg = h[m]
            ag = adj[m][:, m]
            c = max(2, min(self.max_clusters, int(math.ceil(hg.size(0) * self.pool_ratio))))
            sizes.append(c)
            s = F.softmax(self.assign_pool(hg)[:, :c], dim=1)
            s_list.append((m, s))
            n = hg.size(0)
            recon = s @ s.t()
            lp += (ag - recon).norm(p="fro") / n
            ent += -(s * (s + 1e-8).log()).sum() / n

        c_max = max(sizes)
        x2, a2, b2 = [], [], []
        for g, (m, s) in enumerate(s_list):
            pad = torch.zeros(s.size(0), c_max - s.size(1), device=h.device)
            sp = torch.cat([s, pad], dim=1)
            x2.append(sp.t() @ h[m])
            a2.append(sp.t() @ adj[m][:, m] @ sp)
            b2.append(torch.full((c_max,), g, dtype=torch.long, device=h.device))
        x2 = torch.cat(x2, 0)
        a2 = torch.block_diag(*a2)
        b2 = torch.cat(b2, 0)

        h2 = self.conv_after(x2, a2)
        g2 = mean_pool(h2, b2)
        aux = lp / n_g + 0.5 * ent / n_g
        return self.classifier(g2), aux


# The continuous ablation has no VQ quantization, which makes it architecturally and
# numerically identical to the GIN baseline. Keep it as an alias so the "Continuous"
# method label in the benchmark remains distinct without duplicating the class.
ContinuousGIN = GIN


def make_model(name: str, in_dim: int, cfg: dict, num_classes: int) -> nn.Module:
    """Build a model by name."""
    hidden = cfg["hidden"]
    nl = cfg.get("num_layers", 2)
    if name == "GIN":
        return GIN(in_dim, hidden, num_classes, nl)
    if name == "GCN":
        return GCN(in_dim, hidden, num_classes, nl)
    if name == "SAGE":
        return SAGE(in_dim, hidden, num_classes, nl)
    if name == "GAT":
        return GAT(in_dim, hidden, num_classes, nl)
    if name == "DGCNN":
        return DGCNN(in_dim, hidden, num_classes, nl)
    if name == "DiffPool":
        return DiffPool(in_dim, hidden, num_classes, nl)
    if name == "MLP":
        return MLP(in_dim, hidden, num_classes)
    if name == "Ours-VQ":
        return VQGIN(in_dim, hidden, num_classes, cfg["codebook_size"], nl)
    if name == "Continuous":
        return ContinuousGIN(in_dim, hidden, num_classes, nl)
    raise ValueError(f"unknown model {name}")
