from mpmath import im
import os
import torch
import typer
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from ffhq_flow import (
    VAE,
    FFHQDataset,
    build_or_load_latent_cache,
    build_cached_latent_loaders,
    LatentFlowMatchingModule,
    EpochImageLogger,
)
import os
import sys
import json
import shutil
import subprocess
import textwrap
import torch
import typer
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
app = typer.Typer()


def run_training(
    train_dir: str,
    test_dir: str,
    batch_size: int,
    num_workers: int,
    lr: float,
    max_epochs: int,
    vae_ckpt: str,
    latent_cache_dir: str,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae = VAE().to(device)
    if os.path.exists(vae_ckpt):
        try:
            state = torch.load(vae_ckpt, map_location=device)
            vae.load_state_dict(state)
            print(f"Loaded VAE checkpoint from {vae_ckpt}")
        except Exception:
            try:
                vae.load_state_dict(state.get("state_dict", state))
                print(f"Loaded VAE checkpoint (nested state) from {vae_ckpt}")
            except Exception:
                print("Could not load VAE checkpoint; continuing with init weights")
    else:
        print("No VAE checkpoint found; continuing with init weights")

    train_dataset = FFHQDataset(train_dir)
    test_dataset = FFHQDataset(test_dir)

    train_latents, test_latents = build_or_load_latent_cache(
        train_dataset,
        test_dataset,
        vae,
        device,
        cache_dir=latent_cache_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        sample_posterior=True,
        force_rebuild=False,
    )

    train_loader, val_loader = build_cached_latent_loaders(
        train_latents, test_latents, batch_size=batch_size, num_workers=num_workers
    )

    latent_shape = tuple(train_latents.shape[1:])
    module = LatentFlowMatchingModule(latent_shape=latent_shape, lr=lr)

    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join("outputs", "flow_matching", "checkpoints"),
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )
    epoch_logger = EpochImageLogger(
        vae=vae,
        num_samples=8,
        every_n_epochs=1,
        out_dir=os.path.join("outputs", "flow_matching", "samples"),
    )

    trainer_kwargs = dict(
        max_epochs=max_epochs, callbacks=[checkpoint_cb, epoch_logger]
    )
    if torch.cuda.is_available():
        trainer_kwargs.update(dict(accelerator="gpu", devices=1, precision=16))
    else:
        trainer_kwargs.update(dict(accelerator="cpu"))

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)


@app.command()
def main(
    train_dir: str = typer.Option(..., help="Path to training images"),
    test_dir: str = typer.Option(..., help="Path to validation images"),
    batch_size: int = 256,
    num_workers: int = 8,
    lr: float = 8e-4,
    max_epochs: int = 80,
    vae_ckpt: str = os.path.join("outputs", "vae_checkpoint.pth"),
    latent_cache_dir: str = os.path.join("outputs", "latent_cache"),
):
    """Run training locally.
    """
    run_training(
        train_dir,
        test_dir,
        batch_size,
        num_workers,
        lr,
        max_epochs,
        vae_ckpt,
        latent_cache_dir,
    )


if __name__ == "__main__":
    app()
