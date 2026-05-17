import glob
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from PIL import Image

class FFHQDataset(Dataset):
  def __init__(self, root_dir, transform=None):
    self.root_dir = root_dir
    self.transform = transform
    self.image_paths = []
    for ext in ('*.png', '*.jpg', '*.jpeg'):
      self.image_paths.extend(glob.glob(os.path.join(root_dir, ext)))

    self.image_paths.sort()
    print(f'Found {len(self.image_paths)} total images in {root_dir}')

  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self, idx):
    img_path = self.image_paths[idx]
    image = Image.open(img_path).convert('RGB')

    if self.transform:
      image = self.transform(image)
    else:
      default_transform = v2.Compose([
        v2.Resize((256, 256)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
      ])
      image = default_transform(image)

    return image

class CachedLatentDataset(Dataset):
  def __init__(self, latents):
    self.latents = latents

  def __len__(self):
    return self.latents.shape[0]

  def __getitem__(self, index):
    return self.latents[index]


def encode_dataset_to_latents(image_dataset, vae, device, batch_size=64, num_workers=0, sample_posterior=True):
  # DataLoader tuned for throughput
  loader_kwargs = {
    'batch_size': batch_size,
    'shuffle': False,
    'num_workers': num_workers,
    'pin_memory': (device.type == 'cuda'),
    'persistent_workers': (num_workers > 0),
  }
  if num_workers > 0:
    loader_kwargs['prefetch_factor'] = 2
  image_loader = DataLoader(image_dataset, **loader_kwargs)

  latent_batches = []
  vae.to(device)
  vae.eval()
  with torch.no_grad():
    for image_batch in image_loader:
      image_batch = image_batch.to(device, non_blocking=True)
      mu, logvar = vae.encoder(image_batch)
      latent_batch = vae.reparam(mu, logvar) if sample_posterior else mu
      latent_batches.append(latent_batch.cpu())

  return torch.cat(latent_batches, dim=0) if len(latent_batches) > 0 else torch.empty(0)


def build_or_load_latent_cache(
  train_dataset,
  test_dataset,
  vae,
  device,
  cache_dir='outputs/latent_cache',
  batch_size=64,
  num_workers=0,
  sample_posterior=True,
  force_rebuild=False,
):
  cache_path = Path(cache_dir)
  cache_path.mkdir(parents=True, exist_ok=True)

  mode = 'posterior' if sample_posterior else 'mu'
  train_cache_file = cache_path / f'train_latents_{mode}.pt'
  test_cache_file = cache_path / f'test_latents_{mode}.pt'

  if force_rebuild or (not train_cache_file.exists()) or (not test_cache_file.exists()):
    print('Building latent cache...')
    train_latents = encode_dataset_to_latents(
      train_dataset,
      vae,
      device,
      batch_size=batch_size,
      num_workers=num_workers,
      sample_posterior=sample_posterior,
    )
    test_latents = encode_dataset_to_latents(
      test_dataset,
      vae,
      device,
      batch_size=batch_size,
      num_workers=num_workers,
      sample_posterior=sample_posterior,
    )
    torch.save(train_latents, train_cache_file)
    torch.save(test_latents, test_cache_file)
  else:
    print('Loading latent cache from disk...')

  train_latents = torch.load(train_cache_file, map_location='cpu')
  test_latents = torch.load(test_cache_file, map_location='cpu')

  return train_latents, test_latents


def build_cached_latent_loaders(train_latents, test_latents, batch_size=64, num_workers=0):
  train_latent_dataset = CachedLatentDataset(train_latents)
  test_latent_dataset = CachedLatentDataset(test_latents)

  train_kwargs = {
    'batch_size': batch_size,
    'shuffle': True,
    'num_workers': num_workers,
    'pin_memory': torch.cuda.is_available(),
    'persistent_workers': (num_workers > 0),
  }
  if num_workers > 0:
    train_kwargs['prefetch_factor'] = 2
  train_latent_loader = DataLoader(train_latent_dataset, **train_kwargs)

  test_kwargs = {
    'batch_size': batch_size,
    'shuffle': False,
    'num_workers': num_workers,
    'pin_memory': torch.cuda.is_available(),
    'persistent_workers': (num_workers > 0),
  }
  if num_workers > 0:
    test_kwargs['prefetch_factor'] = 2
  test_latent_loader = DataLoader(test_latent_dataset, **test_kwargs)

  return train_latent_loader, test_latent_loader
