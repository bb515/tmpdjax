"""JPEG compression utilities.
This code was ported from https://github.com/bahjat-kawar/ddrm-jpeg/blob/master/functions/jpeg_torch.py

Useful references for understanding this code
https://www.cl.cam.ac.uk/teaching/1011/R08/jpeg/acs10-jpeg.pdf JPEG tutorial Andrew B. Lewis
CUED IIB 4F8: Image Coding 2019-2020 - Lecture 3: The DCT and the JPEG Standard J Lasenby Signal Processing Group, Engineering Department, Cambridge, UK
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax import vjp
from torchvision.datasets import VisionDataset
from typing import Callable, Optional
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from glob import glob
from PIL import Image


def dct1(x):
    """
    Discrete Cosine Transform, Type I
    :param x: the input signal
    :return: the DCT-I of the signal over the last dimension
    """
    x_shape = x.shape
    x = x.reshape(-1, x_shape[-1])
    x = jnp.hstack([x, jnp.flip(x, axis=1)[:, 1:-1]])
    x = jnp.fft.rfft(x).reshape(x_shape)
    return jnp.real(x)


def idct1(x):
    """
    The inverse of DCT-I, which is just a scaled DCT-I
    Our definition if idct1 is such that idct1(dct1(x)) == x
    :param x: the input signal
    :return: the inverse DCT-I of the signal over the last dimension
    """
    n = x.shape[-1]
    return dct1(x) / (2 * (n - 1))


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.reshape(-1, N)
    v = jnp.hstack([x[:, ::2], jnp.flip(x[:, 1::2], axis=1)])
    Vc = jnp.fft.fft(v, axis=1)
    k = -jnp.arange(N, dtype=x.dtype)[None, :] * np.pi / (2 * N)
    W_real = jnp.cos(k)
    W_imag = jnp.sin(k)
    V = jnp.real(Vc) * W_real - jnp.imag(Vc) * W_imag
    if norm == 'ortho':
        V = V.at[:, 0].set(V[:, 0] / (jnp.sqrt(N) * 2))
        V = V.at[:, 1:].set(V[:, 1:] / (jnp.sqrt(N / 2) * 2))

    return 2 * V.reshape(x_shape)


def idct(x, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = x.shape
    N = x_shape[-1]

    X_v = x.reshape(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v = X_v.at[:, 0].set(X_v[:, 0] * jnp.sqrt(N) * 2)
        X_v = X_v.at[:, 1:].set(X_v[:, 1:] * jnp.sqrt(N / 2) * 2)

    k = jnp.arange(x_shape[-1], dtype=x.dtype)[None, :] * np.pi / (2 * N)
    W_real = jnp.cos(k)
    W_imag = jnp.sin(k)
    V_t_real = X_v
    V_t_imag = jnp.hstack([X_v[:, :1] * 0, -jnp.flip(X_v, axis=1)[:, :-1]])
    V_real = V_t_real * W_real - V_t_imag * W_imag
    V_imag = V_t_real * W_imag + V_t_imag * W_real
    V = V_real + 1j * V_imag
    v = jnp.fft.irfft(V, n=V.shape[1], axis=1)
    x = jnp.zeros_like(v)
    x = x.at[:, ::2].set(x[:, ::2] + v[:, :N - (N // 2)])
    x = x.at[:, 1::2].set(x[:, 1::2] + jnp.flip(v, axis=1)[:, :N // 2])
    return x.reshape(x_shape)


def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(0, 2, 1), norm=norm)
    return X2.transpose(0, 2, 1)


def idct_2d(x, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct_2d(dct_2d(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(x, norm=norm)
    print(x1.shape)
    assert 0
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def dct_3d(x, norm=None):
    """
    3-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    print(x.shape)
    assert 0
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    X3 = dct(X2.transpose(-1, -3), norm=norm)
    return X3.transpose(-1, -3).transpose(-1, -2)


def idct_3d(X, norm=None):
    """
    The inverse to 3D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct_3d(dct_3d(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    print(X.shape)
    assert 0
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    x3 = idct(x2.transpose(-1, -3), norm=norm)
    return x3.transpose(-1, -3).transpose(-1, -2)


def get_dct(in_features, _type, norm=None, bias=False):
    """Get any DCT function.

    :param in_features: size of expected input.
    :param type: which dct function in this file to use."""

    # initialise using dct function
    # I = torch.eye(in_features)

    if _type == 'dct1':
        return dct1
        # return lambda x: dct1_test(x, norm=norm)
        # weight.data = dct1(I).data.t()
    elif _type == 'idct1':
        # return lambda x: idct1_test(x, norm=norm)
        return idct1
        # weight.data = idct1(I).data.t()
    elif _type == 'dct':
        return lambda x: dct(x, norm=norm)
        # TODO: does it need transposing here? because of .t()?
        # weight.data = dct(I, norm=norm).data.t()
    elif _type == 'idct':
        # return lambda x: idct_test(x, norm=norm)
        return lambda x: idct(x, norm=norm)
        # weight.data = idct(I, norm=norm).data.t()
    # weight.requires_grad = False # don't learn this!
    # return weight


def apply_linear_2d(x, linear_layer):
    """Can be used with a LinearDCT layer to do a 2D DCT.
    :param x: the input signal
    :param linear_layer: any PyTorch Linear layer
    :return: result of linear layer applied to last 2 dimensions
    """
    X1 = linear_layer(x)
    X2 = linear_layer(X1.transpose(0, 2, 1))
    return X2.transpose(0, 2, 1)


def apply_linear_3d(x, linear_layer):
    """Can be used with a LinearDCT layer to do a 3D DCT.
    :param x: the input signal
    :param linear_layer: any PyTorch Linear layer
    :return: result of linear layer applied to last 3 dimensions
    """
    X1 = linear_layer(x)
    X2 = linear_layer(X1.transpose(-1, -2))
    X3 = linear_layer(X2.transpose(-1, -3))
    return X3.transpose(-1, -3).transpose(-1, -2)


__DATASET__ = {}


def get_dataset(name: str, root: str, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Dataset {name} is not defined.")
    return __DATASET__[name](root=root, **kwargs)


def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = cls
        return cls
    return wrapper


def get_dataloader(dataset: VisionDataset,
                   batch_size: int,
                   num_workers: int,
                   train: bool):
    dataloader = DataLoader(dataset,
                            batch_size,
                            shuffle=train,
                            num_workers=num_workers,
                            drop_last=train)
    return dataloader


@register_dataset(name='ffhq')
class FFHQDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None):
        super().__init__(root, transforms)

        self.fpaths = sorted(glob(root + '/**/*.png', recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')

        if self.transforms is not None:
            img = self.transforms(img)

        return img


def image_grid(x, image_size, num_channels):
    img = x.reshape(-1, image_size, image_size, num_channels)
    w = int(np.sqrt(img.shape[0]))
    img = img[:w**2, :, :, :]
    return img.reshape((w, w, image_size, image_size, num_channels)).transpose((0, 2, 1, 3, 4)).reshape((w * image_size, w * image_size, num_channels))


def get_asset_sample():
  dataset = 'FFHQ'
  batch_size = 4
  transform = transforms.ToTensor()
  dataset = get_dataset(dataset.lower(),
                        root='../assets/',
                        transforms=transform)
  loader = get_dataloader(dataset, batch_size=3, num_workers=0, train=False)
  ref_img = next(iter(loader))
  ref_img = ref_img.detach().cpu().numpy()[2].transpose(1, 2, 0)
  ref_img = np.tile(ref_img, (batch_size, 1, 1, 1))
  return ref_img


def jax_rgb2ycbcr(x):
    """Args: param x: Input signal. Assumes x in range [0, 1] and shape (N, H, W, C)."""
    # Get from [0, 1] to [0, 255]
    x = x * 255
    v = jnp.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = x.dot(v.T)
    ycbcr = ycbcr.at[:, :, :, 1:].set(ycbcr[:, :, :, 1:] + 128)
    return ycbcr


def jax_ycbcr2rgb(x):
    """Args: param x: Input signal. Assumes x in range [0, 1] and shape (N, H, W, C)."""
    v = np.array(
      [[ 1.00000000e+00, -3.68199903e-05,  1.40198758e+00],
       [ 1.00000000e+00, -3.44113281e-01, -7.14103821e-01],
       [ 1.00000000e+00,  1.77197812e+00, -1.34583413e-04]])
    rgb = x.astype(jnp.double)
    rgb = rgb.at[:, :, :, 1:].set(rgb[:, :, :, 1:] - 128.)
    rgb = rgb.dot(v.T)
    return rgb


def chroma_subsample(x):
    """
    Args: param x: Signal with shape (N, H, W, C)."""
    return x[:, :, :, 0:1], x[:, ::2, ::2, 1:]


def general_quant_matrix(quality_factor=10):
    q1 = jnp.array([
        16,  11,  10,  16,  24,  40,  51,  61,
        12,  12,  14,  19,  26,  58,  60,  55,
        14,  13,  16,  24,  40,  57,  69,  56,
        14,  17,  22,  29,  51,  87,  80,  62,
        18,  22,  37,  56,  68, 109, 103,  77,
        24,  35,  55,  64,  81, 104, 113,  92,
        49,  64,  78,  87, 103, 121, 120, 101,
        72,  92,  95,  98, 112, 100, 103,  99
        ])
    q2 = jnp.array([
        17,  18,  24,  47,  99,  99,  99,  99,
        18,  21,  26,  66,  99,  99,  99,  99,
        24,  26,  56,  99,  99,  99,  99,  99,
        47,  66,  99,  99,  99,  99,  99,  99,
        99,  99,  99,  99,  99,  99,  99,  99,
        99,  99,  99,  99,  99,  99,  99,  99,
        99,  99,  99,  99,  99,  99,  99,  99,
        99,  99,  99,  99,  99,  99,  99,  99
    ])
    s = (5000 / quality_factor) if quality_factor < 50 else (200 - 2 * quality_factor)
    q1 = jnp.floor((s * q1 + 50) / 100)
    q1 = jnp.where(q1 < 0, 0, q1)
    q1 = jnp.where(q1 > 255, 255, q1)
    # q1 = q1.at[q1 < 0].set(0)
    # q1 = q1.at[q1 > 255].set(255)
    q2 = jnp.floor((s * q2 + 50) / 100)
    q2 = jnp.where(q2 < 0, 0, q2)
    q2 = jnp.where(q2 > 255, 255, q2)
    # q2 = q2.at[q2 < 0].set(0)
    # q2 = q2.at[q2 > 255].set(255)
    return q1, q2


def quantization_matrix(quality_factor):
    return general_quant_matrix(quality_factor)


def image_to_patches(x):
    return jax.lax.conv_general_dilated_patches(lhs=x, filter_shape=[8, 8], padding='SAME', window_strides=[8, 8])


def get_patches_to_images(shape):
    num_batch, image_size, _, num_channels = shape
    assert num_channels == 3
    x_luma = jnp.ones((num_batch, 1, image_size, image_size))
    x_chroma = jnp.ones((num_batch, 2, image_size//2, image_size//2))
    x_luma, patches_to_image_luma = vjp(
        image_to_patches, x_luma)
    x_chroma, patches_to_image_chroma = vjp(
        image_to_patches, x_chroma)
    return patches_to_image_luma, patches_to_image_chroma


def jpeg_encode(x, quality_factor, patches_to_image_luma, patches_to_image_chroma, shape):
    """
    Args: params:
      x: A batch of size (N x H x W x C) and in [0, 255].
      quality_factor: Quality factor
    """
    # num_batch, image_size, _, num_channels = shape
    x = jax_rgb2ycbcr(x)
    x_luma, x_chroma = chroma_subsample(x)
    # Get x_luma, x_chroma is a batch of size (N x C x H x W)
    x_luma = x_luma.transpose(0, 3, 1, 2)
    x_chroma = x_chroma.transpose(0, 3, 1, 2)

    # github.com/google/jax/discussions/5968
    # Try to achieve same thing as below with reshape and transpose, then do the inverse of that.
    x_luma = image_to_patches(x_luma)
    x_chroma = image_to_patches(x_chroma)

    x_luma = x_luma.reshape(x_luma.shape[0], x_luma.shape[1], -1)
    x_chroma = x_chroma.reshape(x_chroma.shape[0], x_chroma.shape[1], -1)
    x_luma = x_luma.transpose(0, 2, 1)
    x_chroma = x_chroma.transpose(0, 2, 1)
    x_luma = x_luma.reshape(-1, 8, 8) - 128.
    x_chroma = x_chroma.reshape(-1, 8, 8) - 128.
    dct_layer = get_dct(8, 'dct', norm='ortho')
    x_luma = apply_linear_2d(x_luma, dct_layer)
    x_chroma = apply_linear_2d(x_chroma, dct_layer)
    x_luma = x_luma.reshape(-1, 1, 8, 8)
    x_chroma = x_chroma.reshape(-1, 2, 8, 8)
    q1, q2 = quantization_matrix(quality_factor)
    x_luma = x_luma / q1.reshape(1, 8, 8)
    x_chroma = x_chroma / q2.reshape(1, 8, 8)
    x_luma = jnp.rint(x_luma)
    x_chroma = jnp.rint(x_chroma)
    return [x_luma, x_chroma]


def jpeg_decode(x, quality_factor, patches_to_image_luma, patches_to_image_chroma, shape):
    """
    :Args:
        param x: Assume x[0] is a batch of size (N x H//8 x W//8, 1, 8, 8). Assume x[1:] is a batch of size (N x H//8 x W//8, 1, 8, 8)
    """
    x_luma, x_chroma = x
    num_batch, _, image_size, _ = shape
    q1, q2 = quantization_matrix(quality_factor)
    x_luma = x_luma * q1.reshape(1, 8, 8)
    x_chroma = x_chroma * q2.reshape(1, 8, 8)
    x_luma = x_luma.reshape(-1, 8, 8)
    x_chroma = x_chroma.reshape(-1, 8, 8)
    dct_layer = get_dct(8, 'idct', norm='ortho')
    x_luma = apply_linear_2d(x_luma, dct_layer)
    x_chroma = apply_linear_2d(x_chroma, dct_layer)
    x_luma = (x_luma + 128).reshape(num_batch, image_size//8, image_size//8, 64).transpose(0, 3, 1, 2)
    x_chroma = (x_chroma + 128).reshape(num_batch, image_size//16, image_size//16, 64 * 2).transpose(0, 3, 1, 2)
    x_luma = patches_to_image_luma(x_luma)[0]
    x_chroma = patches_to_image_chroma(x_chroma)[0]
    x_chroma_repeated = jnp.zeros((num_batch, 2, image_size, image_size))
    x_chroma_repeated = x_chroma_repeated.at[:, :, 0::2, 0::2].set(x_chroma)
    x_chroma_repeated = x_chroma_repeated.at[:, :, 0::2, 1::2].set(x_chroma)
    x_chroma_repeated = x_chroma_repeated.at[:, :, 1::2, 0::2].set(x_chroma)
    x_chroma_repeated = x_chroma_repeated.at[:, :, 1::2, 1::2].set(x_chroma)
    x = jnp.hstack([x_luma, x_chroma_repeated])
    x = x.transpose(0, 2, 3, 1)
    x = jax_ycbcr2rgb(x)
    # # [0, 255] to [0, 1]
    x = x / 255
    return x


def quantization_encode(x, quality_factor):
    quality_factor = 32
    #to int
    x = (x + 1) / 2
    x = x * 255
    x = x.int()
    # quantize
    x = x // quality_factor
    #to float
    x = x.float()
    x = x / (255/quality_factor)
    x = (x * 2) - 1
    return x


def quantization_decode(x, quality_factor):
    return x

