"""Markov Chains."""
import jax.numpy as jnp
from jax import random, grad, vmap, vjp, jacrev
from diffusionjax.utils import batch_mul, get_timestep
from diffusionjax.solvers import DDIMVP, DDIMVE, SMLD, DDPM


def batch_dot(a, b):
  return vmap(lambda a, b: a.T @ b)(a, b)


class KGDMVP(DDIMVP):
  """Kalman Guided Diffusion Model, Markov chain using the DDIM Markov Chain or VP SDE."""

  def __init__(
    self, y, observation_map, noise_std, shape, model, eta=1.0, beta=None, ts=None
  ):
    super().__init__(model, eta, beta, ts)
    self.estimate_h_x_0_vmap = self.get_estimate_x_0_vmap(observation_map)
    self.analysis_vmap = vmap(self.analysis)
    self.y = y
    self.noise_std = noise_std
    self.num_y = y.shape[1]
    self.observation_map = observation_map
    self.batch_observation_map = vmap(observation_map)
    self.axes_vmap = tuple(range(len(shape) + 1)[1:]) + (0,)

  def analysis(self, y, x, t, timestep, ratio):
    h_x_0, (epsilon, _) = self.estimate_h_x_0_vmap(
      x, t, timestep
    )  # TODO: in python 3.8 this line can be removed by utilizing has_aux=True
    grad_H_x_0 = jacrev(lambda _x: self.estimate_h_x_0_vmap(_x, t, timestep)[0])(x)
    H_grad_H_x_0 = self.batch_observation_map(grad_H_x_0)
    C_yy = ratio * H_grad_H_x_0 + self.noise_std**2 * jnp.eye(self.num_y)
    f = jnp.linalg.solve(C_yy, y - h_x_0)
    ls = grad_H_x_0.transpose(self.axes_vmap) @ f
    return epsilon.squeeze(axis=0), ls

  def posterior(self, x, t):
    timestep = get_timestep(t, self.t0, self.t1, self.num_steps)
    m = self.sqrt_alphas_cumprod[timestep]
    sqrt_1m_alpha = self.sqrt_1m_alphas_cumprod[timestep]
    v = sqrt_1m_alpha**2
    ratio = v / m
    alpha = m**2
    epsilon, ls = self.analysis_vmap(self.y, x, t, timestep, ratio)
    m_prev = self.sqrt_alphas_cumprod_prev[timestep]
    v_prev = self.sqrt_1m_alphas_cumprod_prev[timestep] ** 2
    alpha_prev = m_prev**2
    coeff1 = self.eta * jnp.sqrt((v_prev / v) * (1 - alpha / alpha_prev))
    coeff2 = jnp.sqrt(v_prev - coeff1**2)
    posterior_score = -batch_mul(1.0 / sqrt_1m_alpha, epsilon) + ls
    x_mean = batch_mul(m_prev / m, x) + batch_mul(
      sqrt_1m_alpha * (sqrt_1m_alpha * m_prev / m - coeff2), posterior_score
    )
    std = coeff1
    return x_mean, std


class KGDMVPplus(KGDMVP):
  """KGDMVP with a mask."""

  def analysis(self, y, x, t, timestep, ratio):
    h_x_0, vjp_h_x_0, (epsilon, _) = vjp(
      lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True
    )
    C_yy = (
      ratio * self.observation_map(vjp_h_x_0(self.observation_map(jnp.ones_like(x)))[0])
      + self.noise_std**2
    )
    ls = vjp_h_x_0((y - h_x_0) / C_yy)[0]
    return epsilon.squeeze(axis=0), ls


class KGDMVE(DDIMVE):
  def __init__(
    self, y, observation_map, noise_std, shape, model, eta=1.0, sigma=None, ts=None
  ):
    super().__init__(model, eta, sigma, ts)
    self.eta = eta
    self.model = model
    self.y = y
    self.noise_std = noise_std
    self.num_y = y.shape[1]
    self.estimate_h_x_0_vmap = self.get_estimate_x_0_vmap(observation_map)
    # self.analysis_vmap = vmap(self.analysis)
    self.batch_analysis_vmap = vmap(self.analysis)
    self.observation_map = observation_map
    self.batch_observation_map = vmap(observation_map)
    self.axes_vmap = tuple(range(len(shape) + 1)[1:]) + (0,)

  def analysis(self, y, x, t, timestep, ratio):
    h_x_0, (epsilon, _) = self.estimate_h_x_0_vmap(
      x, t, timestep
    )  # TODO: in python 3.8 this line can be removed by utilizing has_aux=True
    grad_H_x_0 = jacrev(lambda _x: self.estimate_h_x_0_vmap(_x, t, timestep)[0])(x)
    H_grad_H_x_0 = self.batch_observation_map(grad_H_x_0)
    C_yy = ratio * H_grad_H_x_0 + self.noise_std**2 * jnp.eye(self.num_y)
    f = jnp.linalg.solve(C_yy, y - h_x_0)
    ls = grad_H_x_0.transpose(self.axes_vmap) @ f
    return epsilon.squeeze(axis=0), ls

  def posterior(self, x, t):
    timestep = get_timestep(t, self.t0, self.t1, self.num_steps)
    sigma = self.discrete_sigmas[timestep]
    sigma_prev = self.discrete_sigmas_prev[timestep]
    epsilon, ls = self.batch_analysis_vmap(self.y, x, t, timestep, sigma**2)
    coeff1 = self.eta * jnp.sqrt(
      (sigma_prev**2 * (sigma**2 - sigma_prev**2)) / (sigma**2)
    )
    coeff2 = jnp.sqrt(sigma_prev**2 - coeff1**2)
    std = coeff1
    posterior_score = -batch_mul(1.0 / sigma, epsilon) + ls
    x_mean = x + batch_mul(sigma * (sigma - coeff2), posterior_score)
    return x_mean, std


class KGDMVEplus(KGDMVE):
  """KGDMVE with a mask."""

  def analysis(self, y, x, t, timestep, ratio):
    h_x_0, vjp_h_x_0, (epsilon, _) = vjp(
      lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True
    )
    C_yy = (
      ratio * self.observation_map(vjp_h_x_0(self.observation_map(jnp.ones_like(x)))[0])
      + self.noise_std**2
    )
    ls = vjp_h_x_0((y - h_x_0) / C_yy)[0]
    return epsilon.squeeze(axis=0), ls


class PiGDMVP(DDIMVP):
  """PiGDM Song et al. 2023. Markov chain using the DDIM Markov Chain or VP SDE."""

  def __init__(
    self,
    y,
    observation_map,
    noise_std,
    shape,
    model,
    data_variance=1.0,
    eta=1.0,
    beta=None,
    ts=None,
  ):
    super().__init__(model, eta, beta, ts)
    # This method requires clipping in order to remain (robustly, over all samples) numerically stable
    self.estimate_h_x_0_vmap = self.get_estimate_x_0_vmap(
      observation_map, clip=True, centered=True
    )
    self.batch_analysis_vmap = vmap(self.analysis)
    self.y = y
    self.noise_std = noise_std
    self.data_variance = data_variance
    self.observation_map = observation_map

  def analysis(self, y, x, t, timestep, v, alpha):
    h_x_0, vjp_h_x_0, (epsilon, _) = vjp(
      lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True
    )
    # What it should really be set to, following the authors' mathematical reasoning:
    r = v * self.data_variance / (v + alpha * self.data_variance)
    C_yy = 1.0 + self.noise_std**2 / r
    ls = vjp_h_x_0((y - h_x_0) / C_yy)[0]
    return epsilon.squeeze(axis=0), ls

  def posterior(self, x, t):
    timestep = get_timestep(t, self.t0, self.t1, self.num_steps)
    m = self.sqrt_alphas_cumprod[timestep]
    sqrt_1m_alpha = self.sqrt_1m_alphas_cumprod[timestep]
    v = sqrt_1m_alpha**2
    alpha = self.alphas_cumprod[timestep]
    epsilon, ls = self.batch_analysis_vmap(self.y, x, t, timestep, v, alpha)
    m_prev = self.sqrt_alphas_cumprod_prev[timestep]
    v_prev = self.sqrt_1m_alphas_cumprod_prev[timestep] ** 2
    alpha_prev = self.alphas_cumprod_prev[timestep]
    coeff1 = self.eta * jnp.sqrt((v_prev / v) * (1 - alpha / alpha_prev))
    coeff2 = jnp.sqrt(v_prev - coeff1**2)
    # TODO: slightly different to Algorithm 1
    posterior_score = -batch_mul(1.0 / sqrt_1m_alpha, epsilon) + ls
    x_mean = batch_mul(m_prev / m, x) + batch_mul(
      sqrt_1m_alpha * (sqrt_1m_alpha * m_prev / m - coeff2), posterior_score
    )
    std = coeff1
    return x_mean, std


class PiGDMVPplus(PiGDMVP):
  """PiGDMVP with a mask."""

  def analysis(self, y, x, t, timestep, v, alpha):
    h_x_0, vjp_estimate_h_x_0, (epsilon, _) = vjp(
      lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True
    )
    # What it should really be set to, following the authors' mathematical reasoning:
    r = v * self.data_variance / (v + alpha * self.data_variance)
    C_yy = 1.0 + self.noise_std**2 / r
    ls = vjp_estimate_h_x_0((y - h_x_0) / C_yy)[0]
    return epsilon.squeeze(axis=0), ls


class ReproducePiGDMVP(DDIMVP):
  """
  NOTE: We found this method to be unstable on CIFAR10 dataset, even with
    thresholding (clip=True) is used at each step of estimating x_0, and for each weighting
    schedule that we tried.
  PiGDM Song et al. 2023. Markov chain using the DDIM Markov Chain or VP SDE."""

  def __init__(
    self,
    y,
    observation_map,
    noise_std,
    shape,
    model,
    data_variance=1.0,
    eta=1.0,
    beta=None,
    ts=None,
  ):
    super().__init__(model, eta, beta, ts)
    self.data_variance = data_variance
    self.estimate_h_x_0_vmap = self.get_estimate_x_0_vmap(
      observation_map, clip=True, centered=True
    )
    self.batch_analysis_vmap = vmap(self.analysis)
    self.y = y
    self.noise_std = noise_std

  def analysis(self, y, x, t, timestep, v, alpha):
    h_x_0, vjp_estimate_h_x_0, (epsilon, x_0) = vjp(
      lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True
    )
    # Value suggested for VPSDE in original PiGDM paper:
    r = v * self.data_variance / (v + self.data_variance)
    # What it should really be set to, following the authors' mathematical reasoning:
    # r = v * self.data_variance  / (v + alpha * self.data_variance)
    C_yy = 1.0 + self.noise_std**2 / r
    ls = vjp_estimate_h_x_0((y - h_x_0) / C_yy)[0]
    return x_0.squeeze(axis=0), ls, epsilon.squeeze(axis=0)

  def posterior(self, x, t):
    timestep = get_timestep(t, self.t0, self.t1, self.num_steps)
    m = self.sqrt_alphas_cumprod[timestep]
    sqrt_1m_alpha = self.sqrt_1m_alphas_cumprod[timestep]
    v = sqrt_1m_alpha**2
    alpha = self.alphas_cumprod[timestep]
    x_0, ls, epsilon = self.analysis_vmap(self.y, x, t, timestep, v, alpha)
    m_prev = self.sqrt_alphas_cumprod_prev[timestep]
    v_prev = self.sqrt_1m_alphas_cumprod_prev[timestep] ** 2
    alpha_prev = self.alphas_cumprod_prev[timestep]
    coeff1 = self.eta * jnp.sqrt((v_prev / v) * (1 - alpha / alpha_prev))
    coeff2 = jnp.sqrt(v_prev - coeff1**2)
    x_mean = batch_mul(m_prev, x_0) + batch_mul(coeff2, epsilon) + batch_mul(m, ls)
    std = coeff1
    return x_mean, std


class ReproducePiGDMVPplus(ReproducePiGDMVP):
  """
  NOTE: We found this method to be unstable on CIFAR10 dataset, even with
    thresholding (clip=True) is used at each step of estimating x_0, and for each weighting
    schedule that we tried.
  PiGDM with a mask. Song et al. 2023. Markov chain using the DDIM Markov Chain or VP SDE."""

  def analysis(self, y, x, t, timestep, v, alpha):
    h_x_0, vjp_estimate_h_x_0, (epsilon, x_0) = vjp(
      lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True
    )
    # Value suggested for VPSDE in original PiGDM paper:
    r = v * self.data_variance / (v + self.data_variance)
    # What it should really be set to, following the authors' mathematical reasoning:
    # r = v * self.data_variance  / (v + alpha * self.data_variance)
    C_yy = 1.0 + self.noise_std**2 / r
    ls = vjp_estimate_h_x_0((y - h_x_0) / C_yy)[0]
    return x_0.squeeze(axis=0), ls, epsilon.squeeze(axis=0)


class PiGDMVE(DDIMVE):
  """PiGDMVE for the SMLD Markov Chain or VE SDE."""

  def __init__(
    self,
    y,
    observation_map,
    noise_std,
    shape,
    model,
    data_variance=1.0,
    eta=1.0,
    sigma=None,
    ts=None,
  ):
    super().__init__(model, eta, sigma, ts)
    self.y = y
    self.data_variance = data_variance
    self.noise_std = noise_std
    # This method requires clipping in order to remain (robustly, over all samples) numerically stable
    self.estimate_h_x_0_vmap = self.get_estimate_x_0_vmap(
      observation_map, clip=True, centered=False
    )
    self.batch_analysis_vmap = vmap(self.analysis)

  def analysis(self, y, x, t, timestep, v):
    h_x_0, vjp_h_x_0, (epsilon, x_0) = vjp(
      lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True
    )
    r = v * self.data_variance / (v + self.data_variance)
    C_yy = 1.0 + self.noise_std**2 / r
    ls = vjp_h_x_0((y - h_x_0) / C_yy)[0]
    return x_0.squeeze(axis=0), ls, epsilon.squeeze(axis=0)

  def posterior(self, x, t):
    timestep = get_timestep(t, self.t0, self.t1, self.num_steps)
    sigma = self.discrete_sigmas[timestep]
    sigma_prev = self.discrete_sigmas_prev[timestep]
    x_0, ls, epsilon = self.batch_analysis_vmap(self.y, x, t, timestep, sigma**2)
    coeff1 = self.eta * jnp.sqrt(
      (sigma_prev**2 * (sigma**2 - sigma_prev**2)) / (sigma**2)
    )
    coeff2 = jnp.sqrt(sigma_prev**2 - coeff1**2)
    x_mean = x_0 + batch_mul(coeff2, epsilon) + ls
    std = coeff1
    return x_mean, std


class PiGDMVEplus(PiGDMVE):
  """KGDMVE with a mask."""

  def analysis(self, y, x, t, timestep, v):
    h_x_0, vjp_h_x_0, (epsilon, x_0) = vjp(
      lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True
    )
    # Value suggested for VPSDE in original PiGDM paper
    r = v * self.data_variance / (v + self.data_variance)
    C_yy = 1.0 + self.noise_std**2 / r
    ls = vjp_h_x_0((y - h_x_0) / C_yy)[0]
    return x_0.squeeze(axis=0), ls, epsilon.squeeze(axis=0)


class DPSSMLD(SMLD):
  """DPS for SMLD ancestral sampling.
  NOTE: This method requires static thresholding (clip=True) in order to remain
    (robustly, over all samples) numerically stable"""

  def __init__(self, scale, y, observation_map, score, sigma=None, ts=None):
    super().__init__(score, sigma, ts)
    self.y = y
    self.scale = scale
    self.likelihood_score_vmap = self.get_likelihood_score_vmap(
      self.get_estimate_x_0_vmap(observation_map, clip=True, centered=False)
    )

  def get_likelihood_score_vmap(self, estimate_h_x_0_vmap):
    def l2_norm(x, t, timestep, y):
      h_x_0, (s, _) = estimate_h_x_0_vmap(x, t, timestep)
      norm = jnp.linalg.norm(y - h_x_0)
      return norm, s.squeeze(axis=0)

    grad_l2_norm = grad(l2_norm, has_aux=True)
    return vmap(grad_l2_norm)

  def update(self, rng, x, t):
    timestep = get_timestep(t, self.t0, self.t1, self.num_steps)
    ls, score = self.likelihood_score_vmap(x, t, timestep, self.y)
    x_mean, std = self.posterior(score, x, timestep)

    # play around with dps method for the best weighting schedule...
    # sigma = self.discrete_sigmas[timestep]
    # sigma_prev = self.discrete_sigmas_prev[timestep]
    # x_mean = x_mean - batch_mul(1 - sigma_prev**2 / sigma**2, self.scale * ls)
    # x_mean = x_mean - batch_mul(sigma**2, self.scale * ls)
    # Since DPS was empirically derived for VP SDE, the scaling in their paper will not work for VE SDE
    x_mean = x_mean - self.scale * ls  # Not the correct scaling for VE
    z = random.normal(rng, x.shape)
    x = x_mean + batch_mul(std, z)
    return x, x_mean


class DPSDDPM(DDPM):
  """DPS for DDPM ancestral sampling.
  NOTE: This method requires static thresholding (clip=True) in order to remain
    (robustly, over all samples) numerically stable"""

  def __init__(self, scale, y, observation_map, score, beta=None, ts=None):
    super().__init__(score, beta, ts)
    self.y = y
    self.scale = scale
    self.likelihood_score = self.get_likelihood_score(
      self.get_estimate_x_0(observation_map, clip=True, centered=True)
    )
    self.likelihood_score_vmap = self.get_likelihood_score_vmap(
      self.get_estimate_x_0_vmap(observation_map, clip=True, centered=True)
    )

  def get_likelihood_score_vmap(self, estimate_h_x_0_vmap):
    def l2_norm(x, t, timestep, y):
      h_x_0, (s, _) = estimate_h_x_0_vmap(x, t, timestep)
      norm = jnp.linalg.norm(y - h_x_0)
      return norm, s.squeeze(axis=0)

    grad_l2_norm = grad(l2_norm, has_aux=True)
    return vmap(grad_l2_norm)

  def get_likelihood_score(self, estimate_h_x_0):
    batch_norm = vmap(jnp.linalg.norm)

    def l2_norm(x, t, timestep, y):
      h_x_0, (s, _) = estimate_h_x_0(x, t, timestep)
      norm = batch_norm(y - h_x_0)
      norm = jnp.sum(norm)
      return norm, s

    grad_l2_norm = grad(l2_norm, has_aux=True)
    return grad_l2_norm

  def update(self, rng, x, t):
    timestep = get_timestep(t, self.t0, self.t1, self.num_steps)
    likelihood_score, score = self.likelihood_score_vmap(x, t, timestep, self.y)
    x_mean, std = self.posterior(score, x, timestep)
    x_mean -= self.scale * likelihood_score  # DPS
    z = random.normal(rng, x.shape)
    x = x_mean + batch_mul(std, z)
    return x, x_mean


class KPDDPM(DDPM):
  """Kalman posterior for DDPM Ancestral sampling."""

  def __init__(self, y, observation_map, noise_std, shape, score, beta, ts):
    super().__init__(score, beta, ts)
    self.y = y
    self.noise_std = noise_std
    # NOTE: Special case when num_y==1 can be handled correctly by defining observation_map output shape (1,)
    self.num_y = y.shape[1]
    self.shape = shape
    self.estimate_h_x_0_vmap = self.get_estimate_x_0_vmap(observation_map)
    self.analysis_vmap = vmap(self.analysis)
    self.observation_map = observation_map
    self.batch_observation_map = vmap(observation_map)
    self.axes_vmap = tuple(range(len(shape) + 1)[1:]) + (0,)

  def analysis(self, y, x, t, timestep, ratio):
    h_x_0, (_, x_0) = self.estimate_h_x_0_vmap(
      x, t, timestep
    )  # TODO: in python 3.8 this line can be removed by utilizing has_aux=True
    grad_H_x_0 = jacrev(lambda _x: self.estimate_h_x_0_vmap(_x, t, timestep)[0])(x)
    H_grad_H_x_0 = self.batch_observation_map(grad_H_x_0)
    C_yy = H_grad_H_x_0 + self.noise_std**2 / ratio * jnp.eye(self.num_y)
    f = jnp.linalg.solve(C_yy, y - h_x_0)
    ls = grad_H_x_0.transpose(self.axes_vmap) @ f
    return x_0.squeeze(axis=0) + ls

  def posterior(self, x, t):
    timestep = get_timestep(t, self.t0, self.t1, self.num_steps)
    beta = self.discrete_betas[timestep]
    m = self.sqrt_alphas_cumprod[timestep]
    v = self.sqrt_1m_alphas_cumprod[timestep] ** 2
    ratio = v / m
    x_0 = self.analysis_vmap(self.y, x, t, timestep, ratio)
    alpha = self.alphas[timestep]
    m_prev = self.sqrt_alphas_cumprod_prev[timestep]
    v_prev = self.sqrt_1m_alphas_cumprod_prev[timestep] ** 2
    x_mean = batch_mul(jnp.sqrt(alpha) * v_prev / v, x) + batch_mul(
      m_prev * beta / v, x_0
    )
    std = jnp.sqrt(beta * v_prev / v)
    return x_mean, std

  def update(self, rng, x, t):
    x_mean, std = self.posterior(x, t)
    z = random.normal(rng, x.shape)
    x = x_mean + batch_mul(std, z)
    return x, x_mean


class KPDDPMplus(KPDDPM):
  """Kalman posterior for DDPM Ancestral sampling."""

  def analysis(self, y, x, t, timestep, ratio):
    h_x_0, vjp_estimate_h_x_0, (_, x_0) = vjp(
      lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True
    )
    C_yy = (
      self.observation_map(
        vjp_estimate_h_x_0(self.observation_map(jnp.ones_like(x)))[0]
      )
      + self.noise_std**2 / ratio
    )
    ls = vjp_estimate_h_x_0((y - h_x_0) / C_yy)[0]
    return x_0.squeeze(axis=0) + ls


class KPSMLD(SMLD):
  """Kalman posterior for SMLD Ancestral sampling."""

  def __init__(self, y, observation_map, noise_std, shape, score, sigma=None, ts=None):
    super().__init__(score, sigma, ts)
    self.y = y
    self.noise_std = noise_std
    self.num_y = y.shape[1]
    self.estimate_h_x_0_vmap = self.get_estimate_x_0_vmap(
      observation_map, clip=True, centered=False
    )
    self.batch_analysis_vmap = vmap(self.analysis)
    self.observation_map = observation_map
    self.batch_observation_map = vmap(observation_map)
    self.axes_vmap = tuple(range(len(shape) + 1)[1:]) + (0,)

  def analysis(self, y, x, t, timestep, ratio):
    h_x_0, (_, x_0) = self.estimate_h_x_0_vmap(
      x, t, timestep
    )  # TODO: in python 3.8 this line can be removed by utilizing has_aux=True
    grad_H_x_0 = jacrev(lambda _x: self.estimate_h_x_0_vmap(_x, t, timestep)[0])(x)
    H_grad_H_x_0 = self.batch_observation_map(grad_H_x_0)
    C_yy = H_grad_H_x_0 + self.noise_std**2 / ratio * jnp.eye(self.num_y)
    f = jnp.linalg.solve(C_yy, y - h_x_0)
    ls = grad_H_x_0.transpose(self.axes_vmap) @ f
    return x_0.squeeze(axis=0) + ls

  def posterior(self, x, t):
    timestep = get_timestep(t, self.t0, self.t1, self.num_steps)
    sigma = self.discrete_sigmas[timestep]
    sigma_prev = self.discrete_sigmas_prev[timestep]
    x_0 = self.batch_analysis_vmap(self.y, x, t, timestep, sigma**2)
    x_mean = batch_mul(sigma_prev**2 / sigma**2, x) + batch_mul(
      1 - sigma_prev**2 / sigma**2, x_0
    )
    std = jnp.sqrt((sigma_prev**2 * (sigma**2 - sigma_prev**2)) / (sigma**2))
    return x_mean, std, x_0

  def update(self, rng, x, t):
    x_mean, std, x_0 = self.posterior(x, t)
    z = random.normal(rng, x.shape)
    x = x_mean + batch_mul(std, z)
    return x, x_0


class KPSMLDplus(KPSMLD):
  """Kalman posterior for SMLD Ancestral sampling."""

  def analysis(self, y, x, t, timestep, ratio):
    h_x_0, vjp_h_x_0, (_, x_0) = vjp(
      lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True
    )
    C_yy = (
      self.observation_map(vjp_h_x_0(self.observation_map(jnp.ones_like(x)))[0])
      + self.noise_std**2 / ratio
    )
    ls = vjp_h_x_0((y - h_x_0) / C_yy)[0]
    return x_0.squeeze(axis=0) + ls


class KPSMLDdiag(KPSMLD):
  """Kalman posterior for SMLD Ancestral sampling."""

  # def get_grad_estimate_x_0_vmap(self, observation_map):
  #   """https://stackoverflow.com/questions/70956578/jacobian-diagonal-computation-in-jax"""

  #   def estimate_x_0_single(val, i, x, t, timestep):
  #     x_shape = x.shape
  #     x = x.flatten()
  #     x = x.at[i].set(val)
  #     x = x.reshape(x_shape)
  #     x = jnp.expand_dims(x, axis=0)
  #     t = jnp.expand_dims(t, axis=0)
  #     v = self.discrete_sigmas[timestep]**2
  #     s = self.score(x, t)
  #     x_0 = x + v * s
  #     h_x_0 = observation_map(x_0)
  #     return h_x_0[i]
  #   return vmap(value_and_grad(estimate_x_0_single), in_axes=(0, 0, None, None, None))

  def analysis(self, y, x, t, timestep, ratio):
    h_x_0, vjp_h_x_0, (_, x_0) = vjp(
      lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True
    )

    # There is no natural way to do this with JAX's transforms:
    # you cannot map the input, because in general each diagonal entry of the jacobian depends on all inputs.
    # This seems like the best method, but it is too slow for numerical evaluation, and batch size cannot be large (max size tested was one)
    vec_vjp_h_x_0 = vmap(vjp_h_x_0)
    diag = jnp.diag(self.batch_observation_map(vec_vjp_h_x_0(jnp.eye(y.shape[0]))[0]))

    # # This method gives OOM error
    # idx = jnp.arange(len(y))
    # h_x_0, diag = self.grad_estimate_x_0_vmap(x.flatten(), idx, x, t, timestep)
    # diag = self.observation_map(diag)

    # # This method can't be XLA compiled and is way too slow for numerical evaluation
    # diag = jnp.empty(y.shape[0])
    # for i in range(y.shape[0]):
    #   eye = jnp.zeros(y.shape[0])
    #   diag_i = jnp.dot(self.observation_map(vjp_h_x_0(eye)[0]), eye)
    #   diag = diag.at[i].set(diag_i)

    C_yy = diag + self.noise_std**2 / ratio
    ls = vjp_h_x_0((y - h_x_0) / C_yy)[0]
    return x_0.squeeze(axis=0) + ls


class KPDDPMdiag(KPDDPM):
  """Kalman posterior for DDPM Ancestral sampling."""

  def analysis(self, y, x, t, timestep, ratio):
    h_x_0, vjp_h_x_0, (_, x_0) = vjp(
      lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True
    )
    vec_vjp_h_x_0 = vmap(vjp_h_x_0)
    diag = jnp.diag(self.batch_observation_map(vec_vjp_h_x_0(jnp.eye(y.shape[0]))[0]))
    C_yy = diag + self.noise_std**2 / ratio
    ls = vjp_h_x_0((y - h_x_0) / C_yy)[0]
    return x_0.squeeze(axis=0) + ls


class KPDDPMdiagHutchinson(KPDDPM):
  """Kalman posterior for DDPM Ancestral sampling using Hutchinson diagonal estimator."""

  def __init__(self, rng, m, inpainting, adjoint_observation_map, *args, **kwargs):
    """
    Args:
      rng:
      m:
      inpainting: Bool
    """
    super().__init__(*args, **kwargs)
    self.m = m
    self.inpainting = inpainting
    # rng for the Hutchinson estimator
    if rng is None:
      self.rng = random.PRNGKey(42)
    else:
      self.rng = rng
    self.estimate_x_0_vmap = self.get_estimate_x_0_vmap(lambda x: x.flatten())
    self.adjoint_observation_map = adjoint_observation_map
    self.batch_adjoint_observation_map = vmap(adjoint_observation_map)

  def analysis(self, y, x, t, timestep, ratio):
    x_0, vjp_x_0, (_, _) = vjp(
      lambda x: self.estimate_x_0_vmap(x, t, timestep), x, has_aux=True
    )
    h_x_0 = self.observation_map(x_0)
    vec_vjp_x_0 = vmap(vjp_x_0)
    rng, step_rng = random.split(self.rng)  # rng for the Hutchinson estimator
    self.rng = step_rng
    z = random.randint(
      rng,
      shape=(self.m,) + x.shape,
      minval=0,
      maxval=2,
    )
    z = jnp.array(z, dtype=x.dtype) * 2.0 - 1.0
    diag = 1.0 / self.m * jnp.sum(z * vec_vjp_x_0(z)[0], axis=0)
    C_yy = self.batch_observation_map(
      self.batch_observation_map(jnp.diag(diag)).T
    ) + self.noise_std**2 / ratio * jnp.eye(y.shape[0])
    f = jnp.linalg.solve(C_yy, y - h_x_0)
    ls = vjp_x_0(self.adjoint_observation_map(f))[0]
    return x_0 + ls


class KPDDPMdiagHutchinsonplus(KPDDPMdiagHutchinson):
  def analysis(self, y, x, t, timestep, ratio):
    h_x_0, vjp_h_x_0, (_, x_0) = vjp(
      lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True
    )
    # vec_vjp_h_x_0 = vmap(vjp_h_x_0)
    rng, step_rng = random.split(self.rng)  # rng for the Hutchinson estimator
    self.rng = step_rng
    z = random.randint(
      rng,
      shape=(self.m,) + x.shape,
      minval=0,
      maxval=2,
    )
    z = random.normal(rng, shape=(self.m,) + x.shape)
    z = jnp.array(z, dtype=x.dtype) * 2.0 - 1.0
    h_z = self.batch_observation_map(z)
    trace = 1.0 / (self.m * self.num_y) * jnp.sum(self.observation_map(jnp.sum(z * vec_vjp_h_x_0(h_z)[0], axis=0)))

    # NOTE This would be incorrect calculation if h is not a diagonal, since need to calculate diag(H @ diag matrix @ H.T)
    C_yy = trace + self.noise_std**2 / ratio
    ls = vjp_h_x_0((y - h_x_0) / C_yy)[0]
    return x_0.squeeze(axis=0) + ls
