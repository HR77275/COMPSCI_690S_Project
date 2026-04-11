"""Sparse Autoencoder for activation analysis."""

from .model import SparseAutoencoder
from .trainer import SAETrainConfig, train_sae

__all__ = ["SparseAutoencoder", "SAETrainConfig", "train_sae"]
