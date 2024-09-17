from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
import lpips
import numpy as np
import torch

import gc
import tensorflow as tf
import tensorflow_gan as tfgan
import tensorflow_hub as tfhub
import tensorflow_probability as tfp
from tensorflow.image import ssim as tf_ssim
import six

from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import tensorflow as tf
import os

import jax.numpy as jnp


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")
flags.DEFINE_float("noise_std", 0.0, "noise standard")
flags.mark_flags_as_required(["workdir", "config"])

INCEPTION_TFHUB = 'https://tfhub.dev/tensorflow/tfgan/eval/inception/1'
INCEPTION_OUTPUT = 'logits'
INCEPTION_FINAL_POOL = 'pool_3'
_DEFAULT_DTYPES = {
  INCEPTION_OUTPUT: tf.float32,
  INCEPTION_FINAL_POOL: tf.float32
}
INCEPTION_DEFAULT_IMAGE_SIZE = 299


def mse(a, b):
  return np.mean((a - b)**2)


def classifier_fn_from_tfhub(output_fields, inception_model,
                             return_tensor=False):
  """Returns a function that can be as a classifier function.

  Copied from tfgan but avoid loading the model each time calling _classifier_fn

  Args:
    output_fields: A string, list, or `None`. If present, assume the module
      outputs a dictionary, and select this field.
    inception_model: A model loaded from TFHub.
    return_tensor: If `True`, return a single tensor instead of a dictionary.

  Returns:
    A one-argument function that takes an image Tensor and returns outputs.
  """
  if isinstance(output_fields, six.string_types):
    output_fields = [output_fields]

  def _classifier_fn(images):
    output = inception_model(images)
    if output_fields is not None:
      output = {x: output[x] for x in output_fields}
    if return_tensor:
      assert len(output) == 1
      output = list(output.values())[0]
    return tf.nest.map_structure(tf.compat.v1.layers.flatten, output)

  return _classifier_fn


@tf.function
def run_inception_jit(inputs,
                      inception_model,
                      num_batches=1,
                      inceptionv3=False):
  """Running the inception network. Assuming input is within [0, 255]."""
  if not inceptionv3:
    inputs = (tf.cast(inputs, tf.float32) - 127.5) / 127.5
  else:
    inputs = tf.cast(inputs, tf.float32) / 255.

  return tfgan.eval.run_classifier_fn(
    inputs,
    num_batches=num_batches,
    classifier_fn=classifier_fn_from_tfhub(None, inception_model),
    dtypes=_DEFAULT_DTYPES)


@tf.function
def run_inception_distributed(input_tensor,
                              inception_model,
                              num_batches=1,
                              inceptionv3=False):
  """Distribute the inception network computation to all available TPUs.

  Args:
    input_tensor: The input images. Assumed to be within [0, 255].
    inception_model: The inception network model obtained from `tfhub`.
    num_batches: The number of batches used for dividing the input.
    inceptionv3: If `True`, use InceptionV3, otherwise use InceptionV1.

  Returns:
    A dictionary with key `pool_3` and `logits`, representing the pool_3 and
      logits of the inception network respectively.
  """
  num_tpus = 1
  input_tensors = tf.split(input_tensor, num_tpus, axis=0)
  pool3 = []
  logits = [] if not inceptionv3 else None
  device_format = '/GPU:{}'
  for i, tensor in enumerate(input_tensors):
    with tf.device(device_format.format(i)):
      tensor_on_device = tf.identity(tensor)
      res = run_inception_jit(
        tensor_on_device, inception_model, num_batches=num_batches,
        inceptionv3=inceptionv3)

      if not inceptionv3:
        pool3.append(res['pool_3'])
        logits.append(res['logits'])  # pytype: disable=attribute-error
      else:
        pool3.append(res)

  with tf.device('/CPU'):
    return {
      'pool_3': tf.concat(pool3, axis=0),
      'logits': tf.concat(logits, axis=0) if not inceptionv3 else None
    }


def get_inception_model(inceptionv3=False):
  if inceptionv3:
    return tfhub.load(
      'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4')
  else:
    return tfhub.load(INCEPTION_TFHUB)


def load_dataset_stats(dataset):
  """Load the pre-computed dataset statistics."""
  if dataset.lower() == 'cifar10':
    filename = 'assets/cifar10_stats.npz'
  elif dataset == 'ffhq':
    filename = 'assets/ffhq/ffhq_clean_trainval_256.npz'
  else:
    raise ValueError(f'Dataset {dataset} stats not found.')

  with tf.io.gfile.GFile(filename, 'rb') as fin:
    stats = np.load(fin)
    return stats


def compute_metrics_inner(samples, labels):
  samples = np.clip(samples * 255, 0, 255).astype(np.uint8)



def compute_fid_from_stats(samples, dataset, idx=None):
    """https://github.com/tensorflow/gan/blob/656e4332d1e6d7f398f0968966c753e44397fc60/tensorflow_gan/python/eval/classifier_metrics.py#L689"""

    def calculate_fid_helper(activations1, m_w, sigma_w):
        activations1 = tf.convert_to_tensor(value=activations1)
        activations1.shape.assert_has_rank(2)
        m_w = tf.convert_to_tensor(value=m_w)
        m_w.shape.assert_has_rank(1)
        sigma_w = tf.convert_to_tensor(value=sigma_w)
        sigma_w.shape.assert_has_rank(2)

        activations_dtype = activations1.dtype
        if activations_dtype != tf.float64:
            activations1 = tf.cast(activations1, tf.float64)
            m_w = tf.cast(m_w, tf.float64)
            sigma_w = tf.cast(sigma_w, tf.float64)

        m = (tf.reduce_mean(input_tensor=activations1, axis=0),)
        m_w = (m_w,)
        # Calculate the unbiased covariance matrix of first activations.
        num_examples_real = tf.cast(tf.shape(input=activations1)[0], tf.float64)
        sigma = (num_examples_real / (num_examples_real - 1) *
                tfp.stats.covariance(activations1),)
        sigma_w = (sigma_w,)
        # m, m_w, sigma, sigma_w are tuples containing one or two elements: the first
        # element will be used to calculate the score value and the second will be
        # used to create the update_op. We apply the same operation on the two
        # elements to make sure their value is consistent.

        def _symmetric_matrix_square_root(mat, eps=1e-10):
            """Compute square root of a symmetric matrix.

            Note that this is different from an elementwise square root. We want to
            compute M' where M' = sqrt(mat) such that M' * M' = mat.

            Also note that this method **only** works for symmetric matrices.

            Args:
                mat: Matrix to take the square root of.
                eps: Small epsilon such that any element less than eps will not be square
                rooted to guard against numerical instability.

            Returns:
                Matrix square root of mat.
            """
            # Unlike numpy, tensorflow's return order is (s, u, v)
            s, u, v = tf.linalg.svd(mat)
            # sqrt is unstable around 0, just use 0 in such case
            si = tf.compat.v1.where(tf.less(s, eps), s, tf.sqrt(s))
            # Note that the v returned by Tensorflow is v = V
            # (when referencing the equation A = U S V^T)
            # This is unlike Numpy which returns v = V^T
            return tf.matmul(tf.matmul(u, tf.linalg.tensor_diag(si)), v, transpose_b=True)

        def trace_sqrt_product(sigma, sigma_v):
            """Find the trace of the positive sqrt of product of covariance matrices.

            '_symmetric_matrix_square_root' only works for symmetric matrices, so we
            cannot just take _symmetric_matrix_square_root(sigma * sigma_v).
            ('sigma' and 'sigma_v' are symmetric, but their product is not necessarily).

            Let sigma = A A so A = sqrt(sigma), and sigma_v = B B.
            We want to find trace(sqrt(sigma sigma_v)) = trace(sqrt(A A B B))
            Note the following properties:
            (i) forall M1, M2: eigenvalues(M1 M2) = eigenvalues(M2 M1)
                => eigenvalues(A A B B) = eigenvalues (A B B A)
            (ii) if M1 = sqrt(M2), then eigenvalues(M1) = sqrt(eigenvalues(M2))
                => eigenvalues(sqrt(sigma sigma_v)) = sqrt(eigenvalues(A B B A))
            (iii) forall M: trace(M) = sum(eigenvalues(M))
                => trace(sqrt(sigma sigma_v)) = sum(eigenvalues(sqrt(sigma sigma_v)))
                                            = sum(sqrt(eigenvalues(A B B A)))
                                            = sum(eigenvalues(sqrt(A B B A)))
                                            = trace(sqrt(A B B A))
                                            = trace(sqrt(A sigma_v A))
            A = sqrt(sigma). Both sigma and A sigma_v A are symmetric, so we **can**
            use the _symmetric_matrix_square_root function to find the roots of these
            matrices.

            Args:
                sigma: a square, symmetric, real, positive semi-definite covariance matrix
                sigma_v: same as sigma

            Returns:
                The trace of the positive square root of sigma*sigma_v
            """

            # Note sqrt_sigma is called "A" in the proof above
            sqrt_sigma = _symmetric_matrix_square_root(sigma)

            # This is sqrt(A sigma_v A) above
            sqrt_a_sigmav_a = tf.matmul(sqrt_sigma, tf.matmul(sigma_v, sqrt_sigma))

            return tf.linalg.trace(_symmetric_matrix_square_root(sqrt_a_sigmav_a))

        def _calculate_fid(m, m_w, sigma, sigma_w):
            """Returns the Frechet distance given the sample mean and covariance."""
            # Find the Tr(sqrt(sigma sigma_w)) component of FID
            sqrt_trace_component = trace_sqrt_product(sigma, sigma_w)

            # Compute the two components of FID.

            # First the covariance component.
            # Here, note that trace(A + B) = trace(A) + trace(B)
            trace = tf.linalg.trace(sigma + sigma_w) - 2.0 * sqrt_trace_component

            # Next the distance between means.
            mean = tf.reduce_sum(input_tensor=tf.math.squared_difference(
                m, m_w))  # Equivalent to L2 but more stable.
            fid = trace + mean
            if activations_dtype != tf.float64:
                fid = tf.cast(fid, activations_dtype)
            return fid

        result = tuple(
            _calculate_fid(m_val, m_w_val, sigma_val, sigma_w_val)
            for m_val, m_w_val, sigma_val, sigma_w_val in zip(m, m_w, sigma, sigma_w))
        return result[0]

    # Compute FID scores

    uint8_samples = np.clip(samples * 255., 0, 255).astype(np.uint8)
    samples = np.clip(samples, 0., 1.)
    # print("sample range: ", np.min(samples), np.max(samples))
    # print("uint8 sample range: ", np.min(uint8_samples), np.max(uint8_samples))

    # Use inceptionV3 for images with resolution higher than 256.
    # inceptionv3 = image_size >= 256
    inceptionv3 = False
    inception_model = get_inception_model(inceptionv3=inceptionv3)
    # Load pre-computed dataset statistics.
    data_stats = load_dataset_stats(dataset)
    data_mu = data_stats["mu"]
    data_sigma = data_stats["sigma"]

    gc.collect()
    latents = run_inception_distributed(uint8_samples, inception_model, inceptionv3=inceptionv3)
    # Force garbage collection again before returning to JAX code
    gc.collect()

    # tmp_logits = latents["logits"].numpy()
    tmp_pool_3 = latents["pool_3"].numpy()

    # print("tmpd_pool_3.shape: {}".format(tmp_pool_3.shape))
    # must have rank 2 to calculate distribution distances
    assert tmp_pool_3.shape[0] > 1

    # # Compute FID/KID/IS on individual inverse problem
    # if not inceptionv3:
    #     _inception_score = tfgan.eval.classifier_score_from_logits(tmp_logits)
    #     if idx:
    #         stable_inception_score = tfgan.eval.classifier_score_from_logits(tmp_logits[idx])
    # else:
    #     _inception_score = -1

    _fid = calculate_fid_helper(
        tmp_pool_3, data_mu, data_sigma)
    if idx:
        _stable_fid = tfgan.eval.frechet_classifier_distance_from_activations(
            tmp_pool_3[idx], data_mu, data_sigma)
    else: _stable_fid = None

    # # Hack to get tfgan KID work for eager execution.
    # _tf_data_pools = tf.convert_to_tensor(data_pools)
    # _tf_tmp_pools = tf.convert_to_tensor(tmp_pool_3)
    # stable_tf_tmp_pools = tf.convert_to_tensor(tmp_pool_3[idx])
    # _kid = tfgan.eval.kernel_classifier_distance_from_activations(
    #     _tf_data_pools, _tf_tmp_pools).numpy()
    # stable_kid = tfgan.eval.kernel_classifier_distance_from_activations(
    #     _tf_data_pools, stable_tf_tmp_pools).numpy()
    # del _tf_data_pools, _tf_tmp_pools, stable_tf_tmp_pools

    # print(f'{dataset} FID: {_fid}')
    # print(f'{dataset} KID: {_fid}')
    return _fid, _stable_fid


def compute_fid_from_activations(samples, dataset=None, labels=None, idx=None):

    # Compute FID scores
    uint8_samples = np.clip(samples * 255., 0, 255).astype(np.uint8)
    samples = np.clip(samples, 0., 1.)
    # print("sample range: ", np.min(samples), np.max(samples))
    # print("uint8 sample range: ", np.min(uint8_samples), np.max(uint8_samples))

    # Use inceptionV3 for images with resolution higher than 256.
    # inceptionv3 = image_size >= 256
    inceptionv3 = False
    inception_model = get_inception_model(inceptionv3=inceptionv3)
    if dataset is not None:
        # Load pre-computed dataset statistics.
        data_stats = load_dataset_stats(dataset)
        data_pools = data_stats["pool_3"]
    elif dataset is None and labels is not None:
        uint8_labels = np.clip(labels * 255., 0, 255).astype(np.uint8)
        # print("label range: ", np.min(labels), np.max(labels))
        # print("uint8 label range: ", np.min(uint8_labels), np.max(uint8_labels))
        gc.collect()
        data_latents = run_inception_distributed(uint8_labels, inception_model, inceptionv3=inceptionv3)
        # Force garbage collection again before returning to JAX code
        gc.collect()
        data_pools = data_latents["pool_3"]
    else: raise ValueError("must supply dataset statistics or samples")

    gc.collect()
    latents = run_inception_distributed(uint8_samples, inception_model, inceptionv3=inceptionv3)
    # Force garbage collection again before returning to JAX code
    gc.collect()

    tmp_pool_3 = latents["pool_3"].numpy()

    # print("tmp_pool_3.shape: {}".format(tmp_pool_3.shape))
    # must have rank 2 to calculate distribution distances
    assert tmp_pool_3.shape[0] > 1

    # Compute FID/KID/IS on individual inverse problem
    _fid = tfgan.eval.frechet_classifier_distance_from_activations(
        data_pools, tmp_pool_3)
    if idx:
        _stable_fid = tfgan.eval.frechet_classifier_distance_from_activations(
            data_pools, tmp_pool_3[idx])
    else: _stable_fid = None

    # # Hack to get tfgan KID to work for eager execution.
    # _tf_data_pools = tf.convert_to_tensor(data_pools)
    # _tf_tmp_pools = tf.convert_to_tensor(tmp_pool_3)
    # stable_tf_tmp_pools = tf.convert_to_tensor(tmp_pool_3[idx])
    # _kid = tfgan.eval.kernel_classifier_distance_from_activations(
    #     _tf_data_pools, _tf_tmp_pools).numpy()
    # stable_kid = tfgan.eval.kernel_classifier_distance_from_activations(
    #     _tf_data_pools, stable_tf_tmp_pools).numpy()
    # del _tf_data_pools, _tf_tmp_pools, stable_tf_tmp_pools

    # print(f'{dataset} FID: {_fid}')
    # print(f'{dataset} KID: {_fid}')
    return _fid, _stable_fid


def main(argv):
  tf.config.experimental.set_visible_devices([], "GPU")
  os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'  # Less prone to GPU memory fragmentation, which should prevent OOM on CIFAR10
  os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.93'  # preallocate 93 percent of memory, which may cause OOM when the JAX program starts

  cs_methods = ['KPSMLDplus', 'DPSSMLDplus', 'PiGDMVEplus']

  # cs_methods = ['PiGDMVEplus']
  cs_methods = ['KPSMLDplus']

  FLAGS.config.sampling.noise_std = FLAGS.noise_std

  eval_folder = FLAGS.eval_folder 
  config = FLAGS.config

  # eval_file = "{}_{}_{}".format(
  #   config.sampling.noise_std, config.data.dataset, config.sampling.cs_method.lower())  # NEW

  # Use inceptionV3 for images with resolution higher than 256.
  # Load pre-computed dataset statistics.
  for cs_method in cs_methods:
    config.sampling.cs_method = cs_method
    eval_file = "{}_{}_{}".format(
      config.sampling.noise_std, config.data.dataset, config.sampling.cs_method.lower())  # NEW
    print(eval_file)
    # Compute inception scores, FIDs and KIDs.
    # Load all statistics that have been previously computed and saved
    samples = []
    all_lpips = []
    all_psnr = []
    all_ssim = []
    all_mse = []
    stats = tf.io.gfile.glob(os.path.join(eval_folder, eval_file + "_*_stats.npz"))

    print("stats path: {}, length stats: {}".format(
      os.path.join(eval_folder, eval_file + "_*_stats.npz"), len(stats)))

    for stat_file in stats:
      with tf.io.gfile.GFile(stat_file, "rb") as fin:
        stat = np.load(fin)

        tmp_samples = stat["samples"]
        tmp_lpips = stat["lpips"]
        tmp_psnr = stat["psnr"]
        tmp_ssim = stat["ssim"]
        tmp_mse = stat["mse"]

        samples.append(tmp_samples)
        all_lpips.append(tmp_lpips)
        all_psnr.append(tmp_psnr)
        all_mse.append(tmp_mse)
        all_ssim.append(tmp_ssim)

    all_samples = np.concatenate(samples, axis=0)
    all_lpips = np.concatenate(all_lpips, axis=0)
    all_psnr = np.concatenate(all_psnr, axis=0)
    all_ssim = np.concatenate(all_ssim, axis=0)
    all_mse = np.concatenate(all_mse, axis=0)

    lpips_mean = np.mean(all_lpips)
    lpips_std = np.std(all_lpips)
    psnr_mean = np.mean(all_psnr)
    psnr_std = np.std(all_psnr)
    ssim_mean = np.mean(all_ssim)
    ssim_std = np.std(all_ssim)
    mse_mean = np.mean(all_mse)
    mse_std = np.std(all_mse)

    # Find metrics for the subset of images that sampled stably, stable as defined by an mse
    # within a theoretical limit (image has support [0., 1.] so max mse is 1.0)
    idx = np.argwhere(all_mse < 1.0).flatten()  # mse is scalar, so flatten is okay
    fraction_stable = len(idx) / jnp.shape(all_mse)[0]
    all_stable_lpips = all_lpips[idx]
    all_stable_mse = all_mse[idx]
    all_stable_ssim = all_ssim[idx]
    all_stable_psnr = all_psnr[idx]
    stable_lpips_mean = np.mean(all_stable_lpips)
    stable_lpips_std = np.std(all_stable_lpips)
    stable_psnr_mean = np.mean(all_stable_psnr)
    stable_psnr_std = np.std(all_stable_psnr)
    stable_ssim_mean = np.mean(all_stable_ssim)
    stable_ssim_std = np.std(all_stable_ssim)
    stable_mse_mean = np.mean(all_stable_mse)
    stable_mse_std = np.std(all_stable_mse)

    # Compute FID/KID/IS on all samples together.
    # fid, _ = compute_fid_from_activations(all_x, dataset=None, labels=labels_x, idx=None)
    # sfid, _ = compute_fid_from_stats(all_samples[idx], 'ffhq', idx=None)
    fid, _ = compute_fid_from_stats(all_samples, 'ffhq', idx=None)

    print("{} - stable: {}, FID: {:6e}, LPIPS: {:6e}+/-{:3e}, PSNR: {:6e}+/-{:3e}, SSIM: {:6e}+/-{:3e}, MSE: {:6e}+/-{:3e}".format(
        cs_method, fraction_stable, fid,
        lpips_mean, lpips_std, psnr_mean, psnr_std, ssim_mean, ssim_std, mse_mean, mse_std,
        ))
    # print("cs_method-{} - stable: {}, \
    #               FID {:6e}, LPIPS: {:6e}+/-{:3e}, PSNR: {:6e}+/-{:3e}, SSIM: {:6e}+/-{:3e}, MSE: {:6e}+/-{:3e}, \
    #               SLPIPS: {:6e}+/-{:3e}, SPSNR: {:6e}+/-{:3e}, SSSIM: {:6e}+/-{:3e}, SMSE: {:6e}+/-{:3e}".format(
    #     cs_method, fraction_stable, fid,
    #     lpips_mean, lpips_std, psnr_mean, psnr_std, ssim_mean, ssim_std, mse_mean, mse_std,
    #     stable_lpips_mean, stable_lpips_std, stable_psnr_mean, stable_psnr_std, stable_ssim_mean, stable_ssim_std, stable_mse_mean, stable_mse_std,
    #     ))
    np.savez_compressed(
      eval_file + "_reports.npz",
      lpips_mean=lpips_mean, lpips_std=lpips_std,
      psnr_mean=psnr_mean, psnr_std=psnr_std,
      ssim_mean=ssim_mean, ssim_std=ssim_std,
      mse_mean=mse_mean, mse_std=mse_std,
      stable_lpips_mean=stable_lpips_mean, stable_lpips_std=stable_lpips_std,
      stable_psnr=stable_psnr_mean, stable_psnr_std=stable_psnr_std,
      stable_ssim=stable_ssim_mean, stable_ssim_std=stable_ssim_std,
      stable_mse=stable_mse_mean, stable_mse_std=stable_mse_std
      )

if __name__ == "__main__":
    app.run(main)
