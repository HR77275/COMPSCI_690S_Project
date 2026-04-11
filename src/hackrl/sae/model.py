"""Sparse Autoencoder model for learning interpretable feature dictionaries."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SparseAutoencoder(nn.Module):
    """Single-layer sparse autoencoder (encoder: linear+ReLU, decoder: linear).

    The dictionary size is typically 4x-8x the input activation dimension so
    that the learned features are an overcomplete, sparse basis.
    """

    def __init__(self, input_dim: int, dict_size: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.dict_size = dict_size

        self.encoder = nn.Linear(input_dim, dict_size)
        self.decoder = nn.Linear(dict_size, input_dim)

        # Initialization: small random weights
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def encode(self, x: Tensor) -> Tensor:
        """Encode activations into sparse feature space. Returns ReLU codes."""
        return torch.relu(self.encoder(x))

    def decode(self, codes: Tensor) -> Tensor:
        """Reconstruct activations from sparse codes."""
        return self.decoder(codes)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (reconstructed, codes)."""
        codes = self.encode(x)
        reconstructed = self.decode(codes)
        return reconstructed, codes

    def compute_loss(
        self, x: Tensor, *, sparsity_coef: float = 1e-3
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute reconstruction + L1 sparsity loss.

        Returns (total_loss, metrics_dict).
        """
        reconstructed, codes = self.forward(x)
        recon_loss = torch.mean((x - reconstructed) ** 2)
        l1_loss = torch.mean(torch.abs(codes))
        total_loss = recon_loss + sparsity_coef * l1_loss

        # L0: average number of non-zero features per sample
        with torch.no_grad():
            l0 = (codes > 0).float().sum(dim=-1).mean().item()

        return total_loss, {
            "recon_loss": recon_loss.item(),
            "l1_loss": l1_loss.item(),
            "total_loss": total_loss.item(),
            "l0": l0,
        }
