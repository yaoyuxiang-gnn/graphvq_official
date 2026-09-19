"""Shared graph layers and readout for benchmark models.

Contains the GIN/GCN/SAGE convolution layers and the mean-pool readout used by
the graph-classification models and the transfer model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def mean_pool(h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Mean-pool node embeddings per graph.

    Args:
        h: node embeddings ``(N, d)``.
        batch: graph id per node ``(N,)``.

    Returns:
        Graph embeddings ``(num_graphs, d)``.
    """
    n_graphs = int(batch.max().item()) + 1
    out = torch.zeros(n_graphs, h.size(1), device=h.device)
    counts = torch.zeros(n_graphs, device=h.device)
    out.index_add_(0, batch, h)
    counts.index_add_(0, batch, torch.ones(h.size(0), device=h.device))
    return out / counts.clamp(min=1).unsqueeze(1)


def sum_pool(h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Sum-pool node embeddings per graph (the readout used by GIN for counting).

    Args:
        h: node embeddings ``(N, d)``.
        batch: graph id per node ``(N,)``.

    Returns:
        Graph embeddings ``(num_graphs, d)``.
    """
    n_graphs = int(batch.max().item()) + 1
    out = torch.zeros(n_graphs, h.size(1), device=h.device)
    out.index_add_(0, batch, h)
    return out


def code_histogram(idx: torch.Tensor, batch: torch.Tensor, k: int) -> torch.Tensor:
    """Per-graph L2-normalized code-index histogram (WL-style color counting).

    Args:
        idx: assigned code index per node ``(N,)``.
        batch: graph id per node ``(N,)``.
        k: codebook size.

    Returns:
        ``(num_graphs, k)`` count histograms, each row L2-normalized.
    """
    b = int(batch.max().item()) + 1
    flat = batch * k + idx  # collapse (graph, code) into one axis
    cnt = torch.zeros(b * k, device=idx.device)
    cnt.index_add_(0, flat, torch.ones(idx.size(0), device=idx.device))
    cnt = cnt.view(b, k)
    return cnt / cnt.norm(dim=1, keepdim=True).clamp(min=1e-8)


class GINLayer(nn.Module):
    """Graph Isomorphism Network layer (sum aggregation with learnable epsilon)."""

    def __init__(self, in_dim: int, out_dim: int, eps: float = 0.0) -> None:
        super().__init__()
        self.eps = nn.Parameter(torch.tensor(float(eps)))
        self.mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.mlp(adj @ h + self.eps * h)  # adj already has self-loops


class GCNLayer(nn.Module):
    """Graph Convolutional Network layer (symmetric normalization)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.w = nn.Linear(in_dim, out_dim)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        deg = adj.sum(dim=1).clamp(min=1.0)
        d_inv_sqrt = torch.diag(deg ** -0.5)
        norm = d_inv_sqrt @ adj @ d_inv_sqrt
        return F.relu(self.w(norm @ h))


class SAGELayer(nn.Module):
    """GraphSAGE layer (mean neighbour aggregation + self concat)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.w = nn.Linear(2 * in_dim, out_dim)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        deg = adj.sum(dim=1).clamp(min=1.0)
        norm = torch.diag(1.0 / deg) @ adj
        neigh = norm @ h
        return F.relu(self.w(torch.cat([h, neigh], dim=-1)))


class GATLayer(nn.Module):
    """Graph Attention layer (Velickovic et al., ICLR 2018), single attention head.

    Attends only over existing edges (the block-diagonal adjacency already
    contains self-loops, so self-attention is included).
    """

    def __init__(self, in_dim: int, out_dim: int, alpha: float = 0.2) -> None:
        super().__init__()
        self.w = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.zeros(2 * out_dim))
        nn.init.xavier_uniform_(self.a.view(2, out_dim), gain=1.414)
        self.leaky = nn.LeakyReLU(alpha)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        wh = self.w(h)  # (N, D)
        d = wh.size(1)
        # Pairwise attention logits via broadcasting: e_ij = a^T [Wh_i || Wh_j].
        e = wh @ self.a[:d]
        e = self.leaky(e.unsqueeze(0) + (wh @ self.a[d:]).unsqueeze(1))
        e = e.masked_fill(adj <= 0, float("-inf"))
        att = torch.softmax(e, dim=1)
        return F.elu(att @ wh)
