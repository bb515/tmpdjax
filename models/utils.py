"""All functions and modules related to model definition."""
from typing import Any
import flax
import functools
import jax.numpy as jnp
import jax
import numpy as np
from diffusionjax.utils import batch_mul
from diffusionjax.sde import VP, VE


# The dataclass that stores all training states
@flax.struct.dataclass
class State:
  step: int
  optimizer: Any
  lr: float
  model_state: Any
  ema_rate: float
  params_ema: Any
  rng: Any


_MODELS = {}


def register_model(cls=None, *, name=None):
  """A decorator for registering model classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _MODELS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _MODELS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_model(name):
    return _MODELS[name]


def get_sigmas(config):
  """Get sigmas --- the set of noise levels for SMLD from config files.
  Args:
    config: A ConfigDict object parsed from the config file
  Returns:
    sigmas: a jax numpy arrary of noise levels
  """
  sigmas = jnp.exp(
    jnp.linspace(
      jnp.log(config.model.sigma_max), jnp.log(config.model.sigma_min),
      config.model.num_scales))

  return sigmas


def init_model(rng, config, num_devices):
  """Initialize a `flax.linen.Module` model. """
  model_name = config.model.name
  model_def = functools.partial(get_model(model_name), config=config)
  input_shape = (num_devices, config.data.image_size, config.data.image_size, config.data.num_channels)
  label_shape = input_shape[:1]
  fake_input = jnp.zeros(input_shape)
  fake_label = jnp.zeros(label_shape, dtype=jnp.int32)
  params_rng, dropout_rng = jax.random.split(rng)
  model = model_def()
  variables = model.init({'params': params_rng, 'dropout': dropout_rng}, fake_input, fake_label)
  # TODO: BB since model state is deprecated, see how I can load in a model like this with new flax
  # Variables is a `flax.FrozenDict`. It is immutable and respects functional programming
  init_model_state, initial_params = variables.pop('params')
  return model, init_model_state, initial_params


def get_model_fn(model, params, states, train=False):
  """Create a function to give the output of the score-based model.

  Args:
    model: A `flax.linen.Module` object the represent the architecture of score-based model.
    params: A dictionary that contains all trainable parameters.
    states: A dictionary that contains all mutable states.
    train: `True` for training and `False` for evaluation.

  Returns:
    A model function.
  """

  def model_fn(x, labels, rng=None):
    """Compute the output of the score-based model.

    Args:
      x: A mini-batch of input data.
      labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
        for different models.
      rng: If present, it is the random state for dropout

    Returns:
      A tuple of (model output, new mutable states)
    """
    variables = {'params': params, **states}
    if not train:
      return model.apply(variables, x, labels, train=False, mutable=False), states
    else:
      rngs = {'dropout': rng}
      return model.apply(variables, x, labels, train=True, mutable=list(states.keys()), rngs=rngs)
      # if states:
      #   return outputs
      # else:
      #   return outputs, states

  return model_fn


def get_epsilon_fn(sde, model, params, states, train=False, continuous=False, return_state=False):
  """Wraps `epsilon_fn` so that the model output corresponds to a real time-dependent epsilon function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of the score-based model.
    params: A dictionary that contains all trainable parameters.
    states: A dictionary that contains all other mutable parameters.
    train: `True` for training and `False` for evaluation.
    continuous: If `True`, the score-based model is expected to directly take continuous time steps.
    return_state: If `True`, return the new mutable states alongside the model output.

  Returns:
    An epsilon function.
  """
  model_fn = get_model_fn(model, params, states, train=train)

  if isinstance(sde, VP):
    def epsilon_fn(x, t, rng=None):
      # Scale neural network output by standard deviation and flip sign
      if continuous:
        # For VP-trained models, t=0 corresponds to the lowest noise level
        # The maximum value of time embedding is assumed to 999 for
        # continuously-trained models.
        labels = t * 999
        model, state = model_fn(x, labels, rng)
      else:
        # For VP-trained models, t=0 corresponds to the lowest noise level
        labels = t * (sde.N - 1)
        model, state = model_fn(x, labels, rng)

      epsilon = model
      
      if return_state:
        return epsilon, state
      else:
        return epsilon

  elif isinstance(sde, VE):
    def epsilon_fn(x, t, rng=None):
      if continuous:
        labels = sde.std(t)
        std = sde.std(t)
      else:
        # For VE-trained models, t=0 corresponds to the highest noise level
        labels = sde.T - t
        labels *= sde.N - 1
        labels = jnp.round(labels).astype(jnp.int32)
        std = sde.discrete_sigmas[labels.astype(jnp.int32)]

      # TODO: check this
      model, state = model_fn(x, labels, rng)
      epsilon = batch_mul(-std, model)
      if return_state:
        return epsilon, state
      else:
        return epsilon

  else:
    raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  return epsilon_fn


def get_score_fn(sde, model, params, states, train=False, continuous=False, return_state=False):
  """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of the score-based model.
    params: A dictionary that contains all trainable parameters.
    states: A dictionary that contains all other mutable parameters.
    train: `True` for training and `False` for evaluation.
    continuous: If `True`, the score-based model is expected to directly take continuous time steps.
    return_state: If `True`, return the new mutable states alongside the model output.

  Returns:
    A score function.
  """
  model_fn = get_model_fn(model, params, states, train=train)

  if isinstance(sde, VP):
    def score_fn(x, t, rng=None):
      # Scale neural network output by standard deviation and flip sign
      if continuous:
        # For VP-trained models, t=0 corresponds to the lowest noise level
        # The maximum value of time embedding is assumed to 999 for
        # continuously-trained models.
        labels = t * 999
        model, state = model_fn(x, labels, rng)
        std = sde.std(t)
      else:
        # For VP-trained models, t=0 corresponds to the lowest noise level
        labels = t * (sde.N - 1)
        model, state = model_fn(x, labels, rng)
        std = sde.sqrt_1m_alphas_cumprod[labels.astype(jnp.int32)]

      score = batch_mul(-model, 1. / std)
      if return_state:
        return score, state
      else:
        return score

  elif isinstance(sde, VE):
    def score_fn(x, t, rng=None):
      if continuous:
        labels = sde.std(t)
      else:
        # For VE-trained models, t=0 corresponds to the highest noise level
        labels = sde.T - t
        labels *= sde.N - 1
        labels = jnp.round(labels).astype(jnp.int32)

      score, state = model_fn(x, labels, rng)
      if return_state:
        return score, state
      else:
        return score

  else:
    raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  return score_fn


def to_flattened_numpy(x):
  """Flatten a JAX array `x` and convert it to numpy."""
  return np.asarray(x.reshape((-1,)))


def from_flattened_numpy(x, shape):
  """Form a JAX array with the given `shape` from a flattened numpy array `x`."""
  return jnp.asarray(x).reshape(shape)
