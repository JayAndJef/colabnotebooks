import torch
from torch import nn
from einops import rearrange


def sinusoidal_time_embedding(t, embedding_dim, max_period=10_000):
  """
  Encode timestep values into sinusoidal embeddings using sine and cosine functions
  at different frequencies. Used in diffusion models for time-aware conditioning.

  Args:
    t: Timestep values in [0, 1]. Shape: [B] or [B, 1]
    embedding_dim: Output embedding dimension
    max_period: Period of the lowest frequency (default 10,000)

  Returns:
    Sinusoidal embedding tensor of shape [B, embedding_dim]
  """
  if t.dim() == 2:
    t = t.squeeze(-1)

  half_dim = embedding_dim // 2
  freqs = torch.exp(
    -torch.log(torch.tensor(float(max_period), device=t.device))
    * torch.arange(half_dim, device=t.device, dtype=torch.float32)
    / max(half_dim - 1, 1)
  )

  angles = t.float().unsqueeze(1) * freqs.unsqueeze(0)
  emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

  if embedding_dim % 2 == 1:
    emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)

  return emb


class TimeEmbeddingMLP(nn.Module):
  """SiLU-based MLP to project time embeddings to model dimension.

  Example:
    sin_emb = sinusoidal_time_embedding(t, 128)
    proj = TimeEmbeddingMLP(input_dim=128, out_dim=384)
    time_feat = proj(sin_emb)  # shape [B, 384]
  """
  def __init__(self, input_dim=128, out_dim=384, hidden_dim=None):
    super().__init__()
    if hidden_dim is None:
      hidden_dim = max(out_dim, input_dim * 2)

    self.net = nn.Sequential(
      nn.Linear(input_dim, hidden_dim),
      nn.SiLU(),
      nn.Linear(hidden_dim, out_dim),
    )

  def forward(self, x):
    return self.net(x)


def projected_time_embedding(t, sin_dim=128, out_dim=384, max_period=10_000, mlp=None, device=None):
  """Helper: compute sinusoidal embedding then project via provided MLP (or a default one).

  Returns tensor shape [B, out_dim].
  """
  if device is None:
    device = t.device

  sin_emb = sinusoidal_time_embedding(t.to(device), sin_dim, max_period)
  if mlp is None:
    mlp = TimeEmbeddingMLP(input_dim=sin_dim, out_dim=out_dim).to(device)

  return mlp(sin_emb)


class PatchEmbedConv(nn.Module):
  """Conv2d-based patch embedder for latent grids using einops for reshaping.

  Converts `x` of shape [B, in_ch, H, W] into tokens [B, N, embed_dim]
  by applying a Conv2d projection with `kernel_size=patch_size` and
  flattening the spatial dims. No normalization applied here.
  """
  def __init__(self, in_ch=8, embed_dim=384, patch_size=4, stride=2):
    super().__init__()
    self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=stride)

  def forward(self, x):
    # x: [B, in_ch, H, W]
    x = self.proj(x) # [B, embed_dim, H', W']
    x = rearrange(x, 'b d h w -> b (h w) d')
    return x


class QKVProjector(nn.Module):
  """Project input tokens to Q, K, V and return shaped tensors per-head.

  Returns q,k,v each with shape [B, N, num_heads, head_dim].
  """
  def __init__(self, dim, num_heads):
    super().__init__()
    assert dim % num_heads == 0
    self.dim = dim
    self.num_heads = num_heads
    self.head_dim = dim // num_heads
    self.qkv = nn.Linear(dim, dim * 3)

  def forward(self, x):
    # x: [B, N, D]
    B, N, D = x.shape
    qkv = self.qkv(x)  # [B, N, 3*D]
    qkv = qkv.view(B, N, 3, self.num_heads, self.head_dim)
    q = qkv[:, :, 0]
    k = qkv[:, :, 1]
    v = qkv[:, :, 2]
    return q, k, v

class MultiHeadSelfAttentionPlain(nn.Module):
  """Standard multi-head self-attention (no RoPE).

  Inputs:
    x: [B, N, D]
  Output:
    [B, N, D]
  """
  def __init__(self, dim, num_heads=8, dropout=0.0):
    super().__init__()
    assert dim % num_heads == 0
    self.dim = dim
    self.num_heads = num_heads
    self.head_dim = dim // num_heads

    self.qkv_proj = QKVProjector(dim, num_heads)
    self.q_norm = nn.LayerNorm(self.head_dim)
    self.k_norm = nn.LayerNorm(self.head_dim)
    self.out_proj = nn.Linear(dim, dim)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, attn_mask=None):
    B, N, D = x.shape
    q, k, v = self.qkv_proj(x)  # each: [B, N, num_heads, head_dim]

    # [B, heads, N, head_dim]
    q = rearrange(q, 'b n h d -> b h n d')
    k = rearrange(k, 'b n h d -> b h n d')
    v = rearrange(v, 'b n h d -> b h n d')
    
    q = self.q_norm(q)
    k = self.k_norm(k)

    scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
    if attn_mask is not None:
      scores = scores + attn_mask
    attn = torch.softmax(scores, dim=-1)
    attn = self.dropout(attn)

    out = torch.matmul(attn, v)  # [B, heads, N, head_dim]
    out = rearrange(out, 'b h n d -> b n (h d)')

    return self.out_proj(out)


class TransformerBlockPlain(nn.Module):
  """Pre-norm transformer block using the plain attention above.

  Usage:
    block = TransformerBlockPlain(dim=384, num_heads=8)
    out = block(tokens)  # tokens: [B, N, D]
  """
  def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.0):
    super().__init__()
    self.norm1 = nn.LayerNorm(dim)
    self.attn = MultiHeadSelfAttentionPlain(dim, num_heads=num_heads, dropout=dropout)
    self.norm2 = nn.LayerNorm(dim)
    hidden_dim = int(dim * mlp_ratio)
    self.mlp = nn.Sequential(
      nn.Linear(dim, hidden_dim),
      nn.SiLU(),
      nn.Linear(hidden_dim, dim),
      nn.Dropout(dropout),
    )

  def forward(self, x):
    x = x + self.attn(self.norm1(x))
    x = x + self.mlp(self.norm2(x))
    return x


class LearnedPositionalEmbedding(nn.Module):
  """Learned 1D absolute positional embeddings.

  Store a parameter of shape [1, max_positions, dim]. For an input
  token tensor of shape [B, N, D], return the first N positions and
  broadcast for addition.
  """
  def __init__(self, max_positions=1024, dim=384):
    super().__init__()
    self.max_positions = max_positions
    self.dim = dim
    self.pos = nn.Parameter(nn.init.normal_(torch.empty(1, max_positions, dim), std=0.02))

  def forward(self, x):
    # x: [B, N, D]
    N = x.shape[1]
    if N > self.max_positions:
      raise ValueError(f"Requested {N} positions but max is {self.max_positions}")
    return self.pos[:, :N, :].to(x.device)


class ViT(nn.Module):
    """Vision Transformer for latent space modeling.
    
    Architecture:
        - Patch embedding via Conv2d
        - Add learned positional embeddings and timestamp embeddings
        - Stack of transformer blocks with plain multi-head self-attention
    """
    def __init__(
        self,
        latent_w=32,
        latent_h=32,
        patch_size=4,
        stride=2,
        in_channels=8,
        embed_dim=384,
        num_heads=8,
        num_layers=12,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()

        self.latent_w = latent_w
        self.latent_h = latent_h
        self.patch_size = patch_size
        self.stride = stride
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        def _tokens_for_size(size, kernel, stride, padding=0, dilation=1):
            return (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1

        h_tokens = _tokens_for_size(latent_h, patch_size, stride)
        w_tokens = _tokens_for_size(latent_w, patch_size, stride)
        num_tokens = h_tokens * w_tokens
        self.num_tokens = int(num_tokens)

        self.patch_embed = PatchEmbedConv(in_channels, embed_dim, patch_size, stride)
        self.pos_embed = LearnedPositionalEmbedding(max_positions=self.num_tokens, dim=embed_dim)
        self.time_mlp = TimeEmbeddingMLP(input_dim=128, out_dim=embed_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlockPlain(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.spatial_project = nn.ConvTranspose2d(
            in_channels=embed_dim,
            out_channels=in_channels,
            kernel_size=patch_size,
            stride=stride,
        )

    def forward(self, x, t):
        """x: [B, in_channels, latent_w, latent_h]
           t: [B] or [B, 1] with values in [0, 1]
        """

        B, C, H, W = x.shape
        tokens = self.patch_embed(x)  # [B, N, embed_dim]

        if tokens.shape[1] != self.num_tokens:
            raise ValueError(f"Computed tokens ({tokens.shape[1]}) != expected ({self.num_tokens}).\n"
                             "Ensure input spatial size matches ViT latent_w/latent_h and patch settings.")

        pos_emb = self.pos_embed(tokens)  # [1, N, embed_dim]
        time_emb = projected_time_embedding(t, mlp=self.time_mlp, device=x.device).unsqueeze(1)  # [B, 1, embed_dim]
        x = tokens + pos_emb + time_emb  # [B, N, embed_dim]

        for block in self.transformer_blocks:
            x = block(x)  # [B, N, embed_dim]
            
        def _tokens_for_size(size, kernel, stride, padding=0, dilation=1):
            return (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
            
        B, N, D = x.shape
        h_tokens = _tokens_for_size(self.latent_h, self.patch_size, self.stride)
        w_tokens = _tokens_for_size(self.latent_w, self.patch_size, self.stride)
        
        x_map = rearrange(x, 'b (h w) d -> b d h w', h=h_tokens, w=w_tokens)

        out = self.spatial_project(x_map)
        return out
