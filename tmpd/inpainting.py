"""Inpainting utilities."""
import numpy as np
from pathlib import Path
import jax


def get_mask(image_size, mask_name='square'):
    mask_file = Path(__file__).parent / Path(f'masks/{mask_name}.npy')
    mask = np.load(mask_file)
    mask_size = mask.shape[0]
    assert mask_size % image_size == 0
    sub_sample = mask_size // image_size
    mask = mask[::sub_sample, ::sub_sample]
    num_obs = np.count_nonzero(mask)
    mask = np.tile(mask, (3, 1, 1))
    mask = mask.transpose(1, 2, 0)
    mask = mask.flatten()
    num_obs *= 3
    return mask, num_obs


def get_random_mask(rng, image_size, num_channels):
    num_obs = int(image_size)
    idx_obs = jax.random.choice(
    rng, image_size**2, shape=(num_obs,), replace=False)
    mask = np.zeros((image_size**2,), dtype=int)
    mask = mask.at[idx_obs].set(1)
    mask = mask.reshape((image_size, image_size))
    mask = np.tile(mask, (num_channels, 1, 1)).transpose(1, 2, 0)
    num_obs = num_obs * 3  # because of tile
    mask = mask.flatten()
    return mask, num_obs
