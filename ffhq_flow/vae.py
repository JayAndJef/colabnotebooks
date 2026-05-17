import torch
import torch.nn.functional as F
from torch import nn

class SelfAttention(nn.Module):
  def __init__(self, in_channels, num_heads=4):
    super().__init__()
    self.mha = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True)
    self.ln = nn.LayerNorm(in_channels)

  def forward(self, x):
    B, C, H, W = x.shape
    residual = x
    # Reshape (B, C, H, W) to (B, H*W, C)
    x_flat = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
    x_normed = self.ln(x_flat)
    attn, _ = self.mha(x_normed, x_normed, x_normed)
    out = attn.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
    return out + residual

class Encoder(nn.Module):
  def __init__(self, latent_channels=4, in_channels=3):
    super().__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(in_channels, 64, 4, stride=2, padding=1), # 3x256x256 -> 64x128x128
      nn.ReLU(inplace=True),
      nn.Conv2d(64, 128, 4, stride=2, padding=1), # 64x128x128 -> 128x64x64
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 256, 4, stride=2, padding=1), # 128x64x64 -> 256x32x32
      nn.ReLU(inplace=True),
    )
    self.attn = SelfAttention(in_channels=256, num_heads=4)

    self.conv_mu = nn.Conv2d(256, latent_channels, 1)
    self.conv_logvar = nn.Conv2d(256, latent_channels, 1)

  def forward(self, x):
    conv_output = self.conv(x)
    attn_output = self.attn(conv_output)
    mu = self.conv_mu(attn_output)
    logvar = self.conv_logvar(attn_output)
    return mu, logvar

class Decoder(nn.Module):
  def __init__(self, latent_channels=4, out_channels=3):
    super().__init__()
    self.initial_conv = nn.Sequential(
      nn.Conv2d(latent_channels, 256, 3, padding=1),
      nn.ReLU(inplace=True)
    )
    self.attn = SelfAttention(in_channels=256, num_heads=4)

    self.deconv = nn.Sequential(
      # 256x32x32 -> 128x64x64
      nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
      nn.ReLU(inplace=True),

      # 128x64x64 -> 64x128x128
      nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
      nn.ReLU(inplace=True),

      # 64x128x128 -> 3x256x256
      nn.ConvTranspose2d(64, out_channels, 4, stride=2, padding=1),
      nn.Tanh(),
    )

  def forward(self, z):
    x = self.initial_conv(z)
    x = self.attn(x)
    return self.deconv(x)

class VAE(nn.Module):
  def __init__(self, image_channels=3, latent_channels=4):
    super().__init__()
    self.encoder = Encoder(latent_channels, image_channels)
    self.decoder = Decoder(latent_channels, image_channels)

  def reparam(self, mu, logvar):
    """
    reparameterization trick to keep gradients
    """
    std = torch.exp(logvar * 0.5)
    elipson = torch.randn_like(std)
    return mu + elipson * std

  def forward(self, x):
    mu, logvar = self.encoder(x)
    z = self.reparam(mu, logvar)
    return self.decoder(z), mu, logvar


def vae_loss(x_out, x, mu, logvar, percep_loss_fn, beta=1e-4):
  percep_loss = percep_loss_fn(x_out, x).mean() * 0.6
  mse_term = F.mse_loss(x_out, x) * 0.2
  L1_term = F.l1_loss(x_out, x) * 0.8
  kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
  kl_term = kl_loss * beta
  return mse_term + L1_term + kl_term + percep_loss, mse_term, L1_term, kl_term, percep_loss
