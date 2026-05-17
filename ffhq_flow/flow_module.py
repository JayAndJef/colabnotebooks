import os

import torch
import torch.nn.functional as F
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
from torchdiffeq import odeint

from .vit import ViT, projected_time_embedding

class LatentFlowMatchingModule(pl.LightningModule):
  def __init__(self, latent_shape=(8, 32, 32), lr=2e-4, weight_decay=1e-2, num_flow_steps=50):
    super().__init__()
    self.save_hyperparameters()
    self.latent_shape = latent_shape
    self.lr = lr
    self.weight_decay = weight_decay
    self.num_flow_steps = num_flow_steps
    self.flow = ViT(
      latent_w=latent_shape[2],
      latent_h=latent_shape[1],
      patch_size=4,
      stride=2,
      in_channels=latent_shape[0],
      embed_dim=768,
      num_heads=10,
      num_layers=10,
      mlp_ratio=4.0,
      dropout=0.0,
    )

  def flow_matching_loss(self, latents):
    batch_size = latents.shape[0]
    x1 = latents
    x0 = torch.randn_like(x1)
    t = torch.rand(batch_size, device=latents.device)
    t_view = t.view(batch_size, 1, 1, 1)
    x_t = (1.0 - t_view) * x0 + t_view * x1
    target_velocity = x1 - x0
    pred_velocity = self.flow(x_t, t)
    loss = F.mse_loss(pred_velocity, target_velocity)
    return loss

  def training_step(self, batch, batch_idx):
    latents = batch.float()
    loss = self.flow_matching_loss(latents)
    self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=latents.shape[0])
    return loss

  def validation_step(self, batch, batch_idx):
    latents = batch.float()
    loss = self.flow_matching_loss(latents)
    self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=latents.shape[0])
    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    total_epochs = self.trainer.max_epochs if self.trainer is not None else 80
    warmup_epochs = 5
    remaining_epochs = max(total_epochs - warmup_epochs, 1)
    cosine_period = remaining_epochs // 3

    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)
    cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cosine_period, T_mult=1, eta_min=self.lr * 1e-3)
    
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])

    return {
      'optimizer': optimizer,
      'lr_scheduler': {
        'scheduler': scheduler,
        'interval': 'epoch',
      },
    }

  @torch.no_grad()
  def sample_latents(self, num_samples=8, steps=None, solver="dopri5"):
    if steps is None:
      steps = self.num_flow_steps
    sample_shape = (num_samples, *self.latent_shape)
    x0 = torch.randn(sample_shape, device=self.device)

    def ode_func(t, x):
      t_batch = t.expand(x.shape[0]).to(x.device)
      return self.flow(x, t_batch)

    t_eval = torch.linspace(0.0, 1.0, steps + 1, device=self.device)
    traj = odeint(ode_func, x0, t_eval, method=solver)
    return traj[-1]


class EpochImageLogger(pl.Callback):
  def __init__(self, vae, num_samples=8, every_n_epochs=1, out_dir='outputs/flow_matching/samples'):
    super().__init__()
    self.vae = vae
    self.num_samples = num_samples
    self.every_n_epochs = every_n_epochs
    self.out_dir = out_dir

  def on_validation_epoch_end(self, trainer, pl_module):
    if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
      return
    if not trainer.is_global_zero:
      return
    pl_module.eval()
    self.vae.to(pl_module.device)
    self.vae.eval()
    with torch.no_grad():
      latents = pl_module.sample_latents(num_samples=self.num_samples)
      images = self.vae.decoder(latents)
      images = (images.clamp(-1, 1) + 1) / 2
      nrow = min(4, self.num_samples)
      grid = make_grid(images, nrow=nrow)
      os.makedirs(self.out_dir, exist_ok=True)
      path = os.path.join(self.out_dir, f'epoch_{trainer.current_epoch:04d}.png')
      to_pil_image(grid.cpu()).save(path)
      display(to_pil_image(grid.cpu()))
      if trainer.logger is not None:
        experiment = getattr(trainer.logger, 'experiment', None)
        if experiment is not None and hasattr(experiment, 'add_image'):
          experiment.add_image('samples', grid, trainer.current_epoch)
    pl_module.train()
