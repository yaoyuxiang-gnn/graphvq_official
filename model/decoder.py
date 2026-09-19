"""Graph decoder: reconstruct node features from quantized embeddings."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import GraphConvLayer


class GraphDecoder(nn.Module):
    """Reconstruct node features ``x_hat`` from quantized embeddings ``z_q``."""

    def __init__(self, latent_dim: int, hidden_dim: int, out_dim: int, num_layers: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(latent_dim, hidden_dim)
        self.convs = nn.ModuleList([GraphConvLayer(hidden_dim) for _ in range(num_layers)])
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, z_q: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """Decode quantized embeddings back to feature space.

        Args:
            z_q: quantized node embeddings of shape ``(N, latent_dim)``.
            adj_norm: normalized adjacency of shape ``(N, N)``.

        Returns:
            Reconstructed node features of shape ``(N, out_dim)``.
        """
        h = F.relu(self.in_proj(z_q))
        for layer in self.convs:
            h = layer(h, adj_norm)
        return self.out_proj(h)
