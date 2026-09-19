"""Vector quantizer (VQ-VAE style): map continuous embeddings to a shared discrete codebook."""
import torch
import torch.nn as nn


class VectorQuantizer(nn.Module):
    """Discretize embeddings into nearest codebook entries with straight-through gradients.

    Args:
        codebook_size: number of discrete tokens ``K`` (the scaling axis of Seed-10).
        latent_dim: embedding dimension of each token.
    """

    def __init__(self, codebook_size: int, latent_dim: int) -> None:
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(codebook_size, latent_dim) * 0.1)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize embeddings.

        Args:
            z: continuous embeddings of shape ``(..., latent_dim)``.

        Returns:
            ``(z_q, indices, vq_loss)`` where ``z_q`` is the quantized embedding with
            straight-through gradients, ``indices`` are the assigned token ids, and
            ``vq_loss`` combines the commitment loss (updates encoder) and the
            codebook loss (updates the codebook).
        """
        flat = z.reshape(-1, z.size(-1))
        # Squared Euclidean distance between each embedding and each codebook entry.
        dist = (
            (flat ** 2).sum(-1, keepdim=True)
            + (self.codebook ** 2).sum(-1).unsqueeze(0)
            - 2.0 * flat @ self.codebook.t()
        )
        idx = dist.argmin(dim=-1)
        e = self.codebook[idx]  # (N, latent_dim); carries gradient to the codebook.
        z_q = flat + (e - flat).detach()  # straight-through estimator.
        z_q = z_q.reshape(z.shape)
        commitment = ((flat - e.detach()) ** 2).mean()
        codebook_loss = ((e - flat.detach()) ** 2).mean()
        return z_q, idx, commitment + codebook_loss
