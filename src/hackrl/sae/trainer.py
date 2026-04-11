"""Training loop for the Sparse Autoencoder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from hackrl.sae.model import SparseAutoencoder


@dataclass(frozen=True)
class SAETrainConfig:
    """Hyperparameters for SAE training."""

    dict_multiplier: int = 8  # dict_size = input_dim * dict_multiplier
    sparsity_coef: float = 1e-3
    learning_rate: float = 1e-3
    batch_size: int = 256
    epochs: int = 50
    seed: int = 42
    device: str = "cpu"
    checkpoint_dir: str = "artifacts/sae"
    log_interval: int = 5  # log every N epochs


def train_sae(
    activations: Tensor,
    *,
    config: SAETrainConfig | None = None,
) -> tuple[SparseAutoencoder, list[dict[str, float]]]:
    """Train a Sparse Autoencoder on a flat tensor of activations.

    Args:
        activations: shape (N, input_dim) — all timestep activations pooled.
        config: training hyperparameters.

    Returns:
        (trained_sae, history) where history is a list of per-epoch metric dicts.
    """
    config = config or SAETrainConfig()
    torch.manual_seed(config.seed)
    device = torch.device(config.device)

    input_dim = activations.shape[1]
    dict_size = input_dim * config.dict_multiplier

    sae = SparseAutoencoder(input_dim=input_dim, dict_size=dict_size).to(device)
    optimizer = Adam(sae.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    dataset = TensorDataset(activations.to(device))
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)

    history: list[dict[str, float]] = []

    print(
        f"Training SAE: input_dim={input_dim}, dict_size={dict_size}, "
        f"samples={activations.shape[0]}, epochs={config.epochs}"
    )

    for epoch in range(1, config.epochs + 1):
        epoch_recon = 0.0
        epoch_l1 = 0.0
        epoch_l0 = 0.0
        num_batches = 0

        sae.train()
        for (batch,) in loader:
            loss, metrics = sae.compute_loss(batch, sparsity_coef=config.sparsity_coef)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_recon += metrics["recon_loss"]
            epoch_l1 += metrics["l1_loss"]
            epoch_l0 += metrics["l0"]
            num_batches += 1

        scheduler.step()

        avg = {
            "epoch": epoch,
            "recon_loss": epoch_recon / num_batches,
            "l1_loss": epoch_l1 / num_batches,
            "l0": epoch_l0 / num_batches,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(avg)

        if epoch % config.log_interval == 0 or epoch == 1 or epoch == config.epochs:
            print(
                f"  [epoch {epoch:03d}/{config.epochs}] "
                f"recon={avg['recon_loss']:.6f}  l1={avg['l1_loss']:.6f}  "
                f"L0={avg['l0']:.1f}  lr={avg['lr']:.2e}"
            )

    # Save checkpoint
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "sae_trained.pt"
    torch.save(
        {
            "model_state_dict": sae.state_dict(),
            "input_dim": input_dim,
            "dict_size": dict_size,
            "config": {
                "dict_multiplier": config.dict_multiplier,
                "sparsity_coef": config.sparsity_coef,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "seed": config.seed,
            },
            "history": history,
        },
        ckpt_path,
    )
    print(f"Saved SAE checkpoint: {ckpt_path}")

    return sae, history


def load_sae(path: str | Path, *, device: str = "cpu") -> SparseAutoencoder:
    """Load a trained SAE from checkpoint."""
    data = torch.load(path, map_location=device)
    sae = SparseAutoencoder(input_dim=data["input_dim"], dict_size=data["dict_size"])
    sae.load_state_dict(data["model_state_dict"])
    sae.to(device)
    sae.eval()
    return sae
