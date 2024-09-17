import jax.numpy as jnp
from jax import vmap, vjp, random
from diffusionjax.solvers import Solver
from diffusionjax.utils import batch_mul
from diffusionjax.sde import VP
from jax import random, vmap


def get_vjp_guidance_mask_hutchinson_alt(sde, observation_map, y, noise_std, m):
  """
  Uses Hutchinson estimator of diagonal of second moment of the covariance of x_0|x_t.

  Computes two vjps.
  """

  # estimate_h_x_0_vmap = sde.get_estimate_x_0_vmap(observation_map)
  estimate_h_x_0 = sde.get_estimate_x_0(observation_map)
  batch_observation_map = vmap(observation_map)
  batch_batch_observation_map = vmap(vmap(observation_map))

  def guidance_score(rng, x, t):
    h_x_0, vjp_h_x_0, (s, _) = vjp(lambda x: estimate_h_x_0(x, t), x, has_aux=True)
    vec_vjp_h_x_0 = vmap(vjp_h_x_0)
    z = random.randint(
      rng,
      shape=(m,) + x.shape,
      minval=0,
      maxval=2,
    )

    z = jnp.array(z, dtype=x.dtype) * jnp.sqrt(2)
    diag = (
      1.0
      / m
      * jnp.sum(z * vec_vjp_h_x_0(batch_batch_observation_map(z))[0], axis=0)
    )
    diag = batch_observation_map(diag)
    C_yy = sde.ratio(t[0]) * diag + noise_std**2
    innovation = y - h_x_0
    ls = innovation / C_yy
    ls = vjp_h_x_0(ls)[0]
    gs = s + ls
    return gs

  return guidance_score


def get_vjp_guidance_mask_hutchinson(sde, observation_map, y, noise_std, m):
  """
  Uses Hutchinson estimator of diagonal of second moment of the covariance of x_0|x_t.

  Computes two vjps.
  """

  # estimate_h_x_0_vmap = sde.get_estimate_x_0_vmap(observation_map)
  estimate_h_x_0 = sde.get_estimate_x_0(observation_map)
  batch_observation_map = vmap(observation_map)
  batch_batch_observation_map = vmap(vmap(observation_map))

  def guidance_score(rng, x, t):
    h_x_0, vjp_h_x_0, (s, _) = vjp(lambda x: estimate_h_x_0(x, t), x, has_aux=True)
    vec_vjp_h_x_0 = vmap(vjp_h_x_0)
    z = random.randint(
      rng,
      shape=(m,) + x.shape,
      minval=0,
      maxval=2,
    )
    z = jnp.array(z, dtype=x.dtype) * 2.0 - 1.0
    diag = (
      1.0
      / m
      * jnp.sum(z * vec_vjp_h_x_0(batch_batch_observation_map(z))[0], axis=0)
    )
    diag = batch_observation_map(diag)
    C_yy = sde.ratio(t[0]) * diag + noise_std**2
    innovation = y - h_x_0
    ls = innovation / C_yy
    ls = vjp_h_x_0(ls)[0]
    gs = s + ls
    return gs

  return guidance_score


class HutchinsonVP:
  """Variance preserving (VP) SDE, a.k.a. time rescaled Ohnstein Uhlenbeck (OU) SDE."""

  def __init__(self, beta=None, log_mean_coeff=None):
    if beta is None:
      self.beta, self.log_mean_coeff = get_linear_beta_function(
        beta_min=0.1, beta_max=20.0
      )
    else:
      self.beta = beta
      self.log_mean_coeff = log_mean_coeff
    self.beta_min = self.beta(0.0)
    self.beta_max = self.beta(1.0)

  def sde(self, x, t):
    beta_t = self.beta(t)
    drift = -0.5 * batch_mul(beta_t, x)
    diffusion = jnp.sqrt(beta_t)
    return drift, diffusion

  def mean_coeff(self, t):
    return jnp.exp(self.log_mean_coeff(t))

  def std(self, t):
    return jnp.sqrt(self.variance(t))

  def variance(self, t):
    return 1.0 - jnp.exp(2 * self.log_mean_coeff(t))

  def marginal_prob(self, x, t):
    return batch_mul(self.mean_coeff(t), x), jnp.sqrt(self.variance(t))

  def prior(self, rng, shape):
    return random.normal(rng, shape)

  def reverse(self, score):
    fwd_sde = self.sde
    beta = self.beta
    log_mean_coeff = self.log_mean_coeff
    return HutchinsonRVP(score, fwd_sde, beta, log_mean_coeff)

  def r2(self, t, data_variance):
    r"""Analytic variance of the distribution at time zero conditioned on x_t, given crude assumption that
    the data distribution is isotropic-Gaussian.

    .. math::
      \text{Variance of }p_{0}(x_{0}|x_{t}) \text{ if } p_{0}(x_{0}) = \mathcal{N}(0, \text{data_variance}I)
      \text{ and } p_{t|0}(x_{t}|x_{0}) = \mathcal{N}(\sqrt(\alpha_{t})x_0, (1 - \alpha_{t})I)
    """
    alpha = jnp.exp(2 * self.log_mean_coeff(t))
    variance = 1.0 - alpha
    return variance * data_variance / (variance + alpha * data_variance)

  def ratio(self, t):
    """Ratio of marginal variance and mean coeff."""
    return self.variance(t) / self.mean_coeff(t)


class HutchinsonEulerMaruyama(Solver):
  """Euler Maruyama numerical solver of an SDE.
  Functions are designed for a mini-batch of inputs."""

  def __init__(self, sde, ts=None):
    """Constructs an Euler-Maruyama Solver.
    Args:
      sde: A valid SDE class.
    """
    super().__init__(ts)
    self.sde = sde
    self.prior = sde.prior

  def update(self, rng, x, t):
    drift, diffusion = self.sde.sde(rng, x, t)
    f = drift * self.dt
    G = diffusion * jnp.sqrt(self.dt)
    noise = random.normal(rng, x.shape)
    x_mean = x + f
    x = x_mean + batch_mul(G, noise)
    return x, x_mean


class HutchinsonRSDE:
  """Reverse SDE class."""

  def __init__(self, score, forward_sde, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.score = score
    self.forward_sde = forward_sde

  def sde(self, rng, x, t):
    drift, diffusion = self.forward_sde(x, t)
    drift = -drift + batch_mul(diffusion**2, self.score(rng, x, t))
    return drift, diffusion


class HutchinsonRVP(HutchinsonRSDE, VP):
  def get_estimate_x_0_vmap(self, observation_map):
    """
    Get a function returning the MMSE estimate of x_0|x_t.

    Args:
      observation_map: function that operates on unbatched x.
      shape: optional tuple that reshapes x so that it can be operated on.
    """

    def estimate_x_0(x, t):
      x = jnp.expand_dims(x, axis=0)
      t = jnp.expand_dims(t, axis=0)
      m_t = self.mean_coeff(t)
      v_t = self.variance(t)
      s = self.score(x, t)
      x_0 = (x + v_t * s) / m_t
      return observation_map(x_0), (s, x_0)

    return estimate_x_0

  def get_estimate_x_0(self, observation_map, shape=None):
    """
    Get a function returning the MMSE estimate of x_0|x_t.

    Args:
      observation_map: function that operates on unbatched x.
      shape: optional tuple that reshapes x so that it can be operated on.
    """
    batch_observation_map = vmap(observation_map)

    def estimate_x_0(x, t):
      m_t = self.mean_coeff(t)
      v_t = self.variance(t)
      s = self.score(x, t)
      x_0 = batch_mul(x + batch_mul(v_t, s), 1.0 / m_t)
      if shape:
        return batch_observation_map(x_0.reshape(shape)), (s, x_0)
      else:
        return batch_observation_map(x_0), (s, x_0)

    return estimate_x_0

  def correct(self, corrector):
    class CVP(RVP):
      def sde(x, t):
        return corrector(self.score, x, t)

    return CVP(self.score, self.forward_sde, self.beta_min, self.beta_max)

  def guide(self, get_guidance_score, observation_map, *args, **kwargs):
    guidance_score = get_guidance_score(self, observation_map, *args, **kwargs)
    return HutchinsonRVP(
      guidance_score, self.forward_sde, self.beta, self.log_mean_coeff
    )

