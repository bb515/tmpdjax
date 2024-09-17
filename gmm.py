"""Gaussian Mixture Model example."""

from absl import app, flags
from ml_collections.config_flags import config_flags
import jax
from jax import vmap, grad
import jax.random as random
import jax.numpy as jnp
from jax.tree_util import Partial as partial
import numpyro.distributions as dist
import pandas as pd
from tqdm import trange
import torch
from torch import eye, randn_like, vstack, manual_seed
from torch.distributions import MixtureSameFamily, MultivariateNormal, Categorical
from diffusionjax.plot import plot_heatmap
from diffusionjax.utils import get_sampler
from diffusionjax.run_lib import get_ddim_chain
import diffusionjax.sde as sde_lib
from tmpd.samplers import get_cs_sampler
from tmpd.plot import plot_single_image, plot_image
import numpy as np
import logging
import ot
from scipy.stats import wasserstein_distance
import time


logger = logging.getLogger(__name__)


def sliced_wasserstein(dist_1, dist_2, n_slices=100):
  projections = torch.randn(size=(n_slices, dist_1.shape[1]))
  projections = projections / torch.linalg.norm(projections, dim=-1)[:, None]
  dist_1_projected = projections @ dist_1.T
  dist_2_projected = projections @ dist_2.T
  return np.mean(
    [
      wasserstein_distance(u_values=d1.cpu().numpy(), v_values=d2.cpu().numpy())
      for d1, d2 in zip(dist_1_projected, dist_2_projected)
    ]
  )


def get_score_fn(ou_dist, sde):
  """The marginals of the forward process are available in closed form https://arxiv.org/pdf/2308.07983.pdf p5."""
  return vmap(grad(lambda x, t: ou_dist(sde.mean_coeff(t)).log_prob(x)))


def get_model_fn(ou_dist, sde):
  """The marginals of the forward process are available in closed form https://arxiv.org/pdf/2308.07983.pdf p5."""
  return vmap(
    grad(
      lambda x, t: -jnp.sqrt(sde.variance(t)) * ou_dist(sde.mean_coeff(t)).log_prob(x)
    )
  )


def gaussian_posterior(
  y, likelihood_A, likelihood_bias, likelihood_precision, prior_loc, prior_covar
):
  prior_precision_matrix = torch.linalg.inv(prior_covar)
  posterior_precision_matrix = (
    prior_precision_matrix + likelihood_A.T @ likelihood_precision @ likelihood_A
  )
  posterior_covariance_matrix = torch.linalg.inv(posterior_precision_matrix)
  posterior_mean = posterior_covariance_matrix @ (
    likelihood_A.T @ likelihood_precision @ (y - likelihood_bias)
    + prior_precision_matrix @ prior_loc
  )
  try:
    posterior_covariance_matrix = (
      posterior_covariance_matrix + posterior_covariance_matrix.T
    ) / 2
    return MultivariateNormal(
      loc=posterior_mean, covariance_matrix=posterior_covariance_matrix
    )
  except ValueError:
    u, s, v = torch.linalg.svd(posterior_covariance_matrix, full_matrices=False)
    s = s.clip(1e-12, 1e6).real
    posterior_covariance_matrix = u.real @ torch.diag_embed(s) @ v.real
    posterior_covariance_matrix = (
      posterior_covariance_matrix + posterior_covariance_matrix.T
    ) / 2
    return MultivariateNormal(
      loc=posterior_mean, covariance_matrix=posterior_covariance_matrix
    )


def build_extended_svd(A: torch.tensor):
  U, d, V = torch.linalg.svd(A, full_matrices=True)
  coordinate_mask = torch.ones_like(V[0])
  coordinate_mask[len(d) :] = 0
  return U, d, coordinate_mask, V


def ou_mixt(mean_coeff, means, dim, weights):
  cat = Categorical(weights)
  ou_norm = MultivariateNormal(
    vstack(tuple((mean_coeff) * m for m in means)), eye(dim).repeat(len(means), 1, 1)
  )
  return MixtureSameFamily(cat, ou_norm)


def ou_mixt_numpyro(mean_coeff, means, dim, weights):
  means = jnp.vstack(means) * mean_coeff
  covs = jnp.repeat(jnp.eye(dim)[None], axis=0, repeats=means.shape[0])
  return dist.MixtureSameFamily(
    component_distribution=dist.MultivariateNormal(means, covariance_matrix=covs),
    mixing_distribution=dist.Categorical(weights),
  )


def ot_sliced_wasserstein(seed, dist_1, dist_2, n_slices=100):
  return ot.sliced_wasserstein_distance(
    dist_1, dist_2, n_projections=n_slices, seed=seed
  )


def get_posterior(obs, prior, A, Sigma_y):
  modified_means = []
  modified_covars = []
  weights = []
  precision = torch.linalg.inv(Sigma_y)
  for loc, cov, weight in zip(
    prior.component_distribution.loc,
    prior.component_distribution.covariance_matrix,
    prior.mixture_distribution.probs,
  ):
    new_dist = gaussian_posterior(obs, A, torch.zeros_like(obs), precision, loc, cov)
    modified_means.append(new_dist.loc)
    modified_covars.append(new_dist.covariance_matrix)
    prior_x = MultivariateNormal(loc=loc, covariance_matrix=cov)
    residue = obs - A @ new_dist.loc
    log_constant = (
      -(residue[None, :] @ precision @ residue[:, None]) / 2
      + prior_x.log_prob(new_dist.loc)
      - new_dist.log_prob(new_dist.loc)
    )
    weights.append(torch.log(weight).item() + log_constant)
  weights = torch.tensor(weights)
  weights = weights - torch.logsumexp(weights, dim=0)
  cat = Categorical(logits=weights)
  ou_norm = MultivariateNormal(
    loc=torch.stack(modified_means, dim=0),
    covariance_matrix=torch.stack(modified_covars, dim=0),
  )
  return MixtureSameFamily(cat, ou_norm)


def generate_measurement_equations(dim, dim_y, device, mixt, noise_std):
  A = torch.randn((dim_y, dim))
  u, diag, coordinate_mask, v = build_extended_svd(A)
  diag = torch.sort(torch.rand_like(diag), descending=True).values
  A = u @ (torch.diag(diag) @ v[coordinate_mask == 1, :])
  init_sample = mixt.sample()
  init_obs = A @ init_sample
  init_obs += randn_like(init_obs) * noise_std
  Sigma_y = torch.diag(noise_std**2 * torch.ones(len(diag)))
  posterior = get_posterior(init_obs, mixt, A, Sigma_y)
  return A, Sigma_y, u, diag, coordinate_mask, v, posterior, init_obs


def calculate_distances(
    config, sde, fn, y, H, observation_map, adjoint_observation_map,
    sample_rng, num_devices, size, 
    dim_y,
    seed_num_inv_problem,
    posterior_samples,
    cs_method):
  config.sampling.cs_method = cs_method
  sampler = get_cs_sampler(
    config,
    sde,
    fn,
    (config.eval.batch_size // num_devices, config.data.image_size),
    None,  # dataset.get_data_inverse_scaler(config),
    y,
    H,
    observation_map,
    adjoint_observation_map,
    stack_samples=config.sampling.stack_samples,
  )
  time_prev = time.time()
  samples, _ = sampler(sample_rng)
  sample_time = time.time() - time_prev

  if config.sampling.stack_samples:
    samples = samples.reshape(
      config.solver.num_outer_steps,
      config.eval.batch_size,
      config.data.image_size,
    )
    plot_heatmap(
      samples=samples[0],
      area_min=-size * 2.5,
      area_max=size * 2.5,
      fname="{} heatmap conditional".format(config.sampling.cs_method),
    )
  else:
    samples = samples.reshape(config.eval.batch_size, config.data.image_size)
    sliced_wasserstein_distance = sliced_wasserstein(
      dist_1=np.array(posterior_samples),
      dist_2=np.array(samples),
      n_slices=10000,
    )
    ot_sliced_wasserstein_distance = ot_sliced_wasserstein(
      seed=seed_num_inv_problem,
      dist_1=np.array(posterior_samples),
      dist_2=np.array(samples),
      n_slices=10000,
    )
    print(
      "sample_time: {}, {}".format(sample_time, config.sampling.cs_method),
      sliced_wasserstein_distance,
      ot_sliced_wasserstein_distance,
    )
    return samples, {
        "seed": seed_num_inv_problem,
        "dim": config.data.image_size,
        "dim_y": dim_y,
        "noise_std": config.sampling.noise_std,
        "num_steps": config.solver.num_outer_steps,
        "algorithm": config.sampling.cs_method,
        "distance_name": "sw",
        "distance": sliced_wasserstein_distance,
        "ot_distance": ot_sliced_wasserstein_distance,
      }


def gmm_experiment(workdir, config, num_devices):
  color_posterior = "#a2c4c9"
  color_algorithm = "#ff7878"

  # Torch device
  device = "cpu"
  dists_infos = []

  # Setup SDE
  if config.training.sde.lower() == "vpsde":
    from diffusionjax.utils import get_linear_beta_function

    beta, log_mean_coeff = get_linear_beta_function(
      config.model.beta_min, config.model.beta_max
    )
    sde = sde_lib.VP(beta, log_mean_coeff)
  else:
    raise NotImplementedError(f"SDE {config.training.SDE} unknown.")

  ind_dim = 0
  ind_increase = 0
  size = 8.0
  num_repeats = 20

  for ind_dim, dim in enumerate([8, 80, 800]):
    config.data.image_size = dim

    # setup of the inverse problem
    means = []
    for i in range(-2, 3):
      means += [
        torch.tensor([-size * i, -size * j] * (config.data.image_size // 2)).to(device)
        for j in range(-2, 3)
      ]
    weights = torch.ones(len(means))
    weights = weights / weights.sum()
    ou_mixt_fun = partial(
      ou_mixt, means=means, dim=config.data.image_size, weights=weights
    )
    ou_mixt_jax_fun = partial(
      ou_mixt_numpyro,
      means=[jnp.array(m.numpy()) for m in means],
      dim=config.data.image_size,
      weights=jnp.array(weights.numpy()),
    )

    rng = random.PRNGKey(config.seed)
    mixt_jax = ou_mixt_jax_fun(1)
    target_samples = mixt_jax.sample(rng, (config.eval.batch_size,))
    # plot_samples(target_samples, index=(0, 1), fname="target gmm jax")

    mixt = ou_mixt_fun(1)
    target_samples = mixt.sample((config.eval.batch_size,))
    logging.info(
      "target prior:\nmean {},\nvar {}".format(
        np.mean(target_samples.numpy(), axis=0), np.var(target_samples.numpy(), axis=0)
      )
    )

    # Plot prior samples
    score = get_score_fn(ou_mixt_jax_fun, sde)
    model = get_model_fn(ou_mixt_jax_fun, sde)

    # outer_solver = get_markov_chain(config, score)
    outer_solver = get_ddim_chain(config, model)
    inner_solver = None

    sampling_shape = (config.eval.batch_size // num_devices, config.data.image_size)
    sampler = get_sampler(
      sampling_shape,
      outer_solver,
      inner_solver,
      denoise=config.sampling.denoise,
      stack_samples=False,
    )
    rng, sample_rng = random.split(rng, 2)
    samples, nfe = sampler(sample_rng)
    logging.info(
      "diffusion prior:\nmean {},\nvar {}".format(
        np.mean(samples, axis=0), np.var(samples, axis=0)
      )
    )
    plot_single_image(
      config.sampling.noise_std,
      dim,
      "_",
      1000,
      0,
      "prior",
      [0, 1],
      samples,
      color=color_algorithm,
    )

    for ind_ptg, dim_y in enumerate([4, 2, 1]):
      for i in trange(0, num_repeats, unit="trials dim_y={}".format(dim_y)):
        seed_num_inv_problem = (2 ** (ind_dim)) * (
          3 ** (ind_ptg) * (5 ** (ind_increase))
        ) + i
        manual_seed(seed_num_inv_problem)
        A, _, _, _, _, _, posterior, init_obs = (
          generate_measurement_equations(
            config.data.image_size, dim_y, device, mixt, config.sampling.noise_std
          )
        )
        # config.sampling.noise_std = float(Sigma_y.numpy()[0, 0])
        logging.info(
          "ind_ptg {:d}, dim {:d}, dim_y {:d}, trial {:d}, noise_std {:.2e}".format(
            ind_ptg, config.data.image_size, dim_y, i, config.sampling.noise_std
          )
        )

        # Getting posterior samples form nuts
        posterior_samples_torch = posterior.sample((config.eval.batch_size,)).to(device)
        posterior_samples = posterior_samples_torch.numpy()
        plot_single_image(
          config.sampling.noise_std,
          dim,
          dim_y,
          1000,
          i,
          "posterior",
          [0, 1],
          posterior_samples,
          color=color_posterior,
        )
        y = jnp.array(init_obs.numpy(), dtype=jnp.float32)
        y = jnp.tile(y, (config.eval.batch_size // num_devices, 1))
        H = jnp.array(A.numpy(), dtype=jnp.float32)

        def observation_map(x):
          x = x.flatten()
          return H @ x

        def adjoint_observation_map(y):
          y = y.flatten()
          return H.T @ y

        ddim_methods = [
          "PiGDMVP",
          "PiGDMVE",
          "DDIMVE",
          "DDIMVP",
          "KGDMVP",
          "KGDMVE",
          "STSL",
        ]
        cs_methods = ['TMPD2023avjp', 'TMPD2023bvjp', 'Chung2022scalar', 'Song2023']

        for cs_method in cs_methods:
          fn = model if cs_method in ddim_methods else score
          dist_info = calculate_distances(
            config, sde, fn, y, H, observation_map, adjoint_observation_map,
            sample_rng, num_devices, size, 
            dim_y,
            seed_num_inv_problem,
            posterior_samples,
            cs_method)
          dists_infos.append(
            dist_info
          )
          plot_image(
            config.sampling.noise_std,
            dim,
            dim_y,
            1000,
            i,
            cs_method,
            [0, 1],
            samples,
            posterior_samples,
          )

        pd.DataFrame.from_records(dists_infos).to_csv(
          workdir
          + "/{}_{}_gmm_inverse_problem_comparison.csv".format(
            config.sampling.cs_method, config.sampling.noise_std
          ),
          float_format="%.3f",
        )

      data = pd.read_csv(
        workdir
        + "/{}_{}_gmm_inverse_problem_comparison.csv".format(
          config.sampling.cs_method, config.sampling.noise_std
        )
      )
      agg_data = (
        data.groupby(["dim", "dim_y", "num_steps", "algorithm", "distance_name"])
        .distance.apply(
          lambda x: f"{np.nanmean(x):.1f} ± {1.96 * np.nanstd(x) / (x.shape[0]**.5):.1f}"
        )
        .reset_index()
      )

      agg_data_sw = (
        agg_data.loc[agg_data.distance_name == "sw"]
        .pivot(
          index=("dim", "dim_y", "num_steps"), columns="algorithm", values=["distance"]
        )
        .reset_index()
      )
      agg_data_sw.columns = [
        col[-1].replace(" ", "_") if col[-1] else col[0].replace(" ", "_")
        for col in agg_data_sw.columns.values
      ]

      for col in agg_data_sw.columns:
        if col not in ["dim", "dim_y", "num_steps"]:
          agg_data_sw[col + "_num"] = agg_data_sw[col].apply(
            lambda x: float(x.split(" ± ")[0])
          )
      agg_data_sw.loc[agg_data_sw.num_steps == 1000].to_csv(
        workdir
        + "/{}_{}_gmm_inverse_problem_aggregated_sliced_wasserstein_1000_steps.csv".format(
          config.sampling.cs_method, config.sampling.noise_std
        )
      )


def main(argv):
  FLAGS = flags.FLAGS
  config_flags.DEFINE_config_file(
    "config", "./configs/gmm.py", "Training configuration.", lock_config=True
  )
  flags.DEFINE_string("workdir", "./workdir", "Work directory.")
  flags.mark_flags_as_required(["workdir", "config"])
  config = FLAGS.config
  workdir = FLAGS.workdir
  jax.default_device = jax.devices()[0]
  # Tip: use CUDA_VISIBLE_DEVICES to restrict the devices visible to jax
  # ... they must be all the same model of device for pmap to work
  num_devices = int(jax.local_device_count()) if config.training.pmap else 1
  gmm_experiment(workdir, config, num_devices)


if __name__ == "__main__":
  app.run(main)

