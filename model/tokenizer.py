"""Graph VQ tokenizer: encode -> quantize -> reconstruct (the Seed-6 + Seed-10 core model)."""
import torch
import torch.nn as nn

from .encoder import GraphEncoder
from .decoder import GraphDecoder
from .vector_quantizer import VectorQuantizer


class GraphVQTokenizer(nn.Module):
    """VQ-VAE style graph autoencoder with a shared discrete codebook.

    The codebook size ``K`` is the independent scaling axis studied by Seed-10;
    the encoder/decoder realize the graph tokenization of Seed-6.
    """

    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int, num_layers: int, codebook_size: int) -> None:
        super().__init__()
        self.encoder = GraphEncoder(in_dim, hidden_dim, latent_dim, num_layers)
        self.quantizer = VectorQuantizer(codebook_size, latent_dim)
        self.decoder = GraphDecoder(latent_dim, hidden_dim, in_dim, num_layers)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode, quantize and reconstruct.

        Returns:
            ``(x_hat, z_q, vq_loss)``.
        """
        z = self.encoder(x, adj_norm)
        z_q, _, vq_loss = self.quantizer(z)
        x_hat = self.decoder(z_q, adj_norm)
        return x_hat, z_q, vq_loss
