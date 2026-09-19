"""Graph encoder: a message-passing GNN mapping node features to node embeddings."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    """A single graph-convolution layer over a (symmetrically) normalized adjacency.

    Args:
        dim: feature dimension of the layer.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.w = nn.Linear(dim, dim)

    def forward(self, h: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """Aggregate 1-hop neighbours and apply a learned linear map + ReLU."""
        msg = adj_norm @ h
        return F.relu(self.w(msg))


class GraphEncoder(nn.Module):
    """Stacked message-passing encoder: ``(x, A) -> z`` (node embeddings)."""

    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int, num_layers: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList([GraphConvLayer(hidden_dim) for _ in range(num_layers)])
        self.out_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """Encode node features.

        Args:
            x: node features of shape ``(N, in_dim)``.
            adj_norm: normalized adjacency of shape ``(N, N)``.

        Returns:
            Node embeddings of shape ``(N, latent_dim)``.
        """
        h = F.relu(self.in_proj(x))
        for layer in self.convs:
            h = layer(h, adj_norm)
        return self.out_proj(h)
