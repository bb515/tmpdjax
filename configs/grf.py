"""Config for `grf.py`."""

from configs.default_cs_configs import get_default_configs


def get_config():
  config = get_default_configs()
  training = config.training
  model = config.model
  eval = config.eval
  sampling = config.sampling
  data = config.data
  solver = config.solver

  # sampling.cs_methods = "song2023plus"
  # sampling.cs_method = "tmpd2023ajacfwd"
  # sampling.cs_method = "tmpd2023ajacrev"
  sampling.cs_method = "tmpd2023avjp"
  # sampling.cs_method = "chung2022scalarplus"
  # sampling.cs_method = "tmpd2023b"
  # sampling.cs_method = "tmpd2023bvjpplus"
  # sampling.cs_method = "tmpd2023bjacfwd"
  # sampling.cs_method = "tmpd2023bvjp"
  # sampling.cs_method = "kpsmldplus"
  # sampling.cs_method = "kgdmvp"
  # sampling.cs_method = "kgdmvpplus"
  # sampling.cs_method = "kgdmve"
  # sampling.cs_method = "kgdmveplus"
  # sampling.cs_method = "pigdmvp"
  # sampling.cs_method = "pigdmvpplus"
  # sampling.cs_method = "pigdmve"
  # sampling.cs_method = "pigdmveplus"
  # sampling.cs_method = "dpssmld"
  # sampling.cs_method = "dpssmldplus"
  # sampling.cs_method = "kpddpm"
  # sampling.cs_method = "kpddpmplus"
  # sampling.cs_method = "kpsmld"
  # sampling.cs_method = "kpsmldplus"

  eval.pmap = True
  data.image_size = 32
  data.num_channels = 1
  eval.batch_size = 1

  training.sde = "vpsde"
  training.num_epochs = 4000
  training.batch_size = 16

  sampling.noise_std = 0.1
  sampling.denoise = True  # work out what denoise_override is
  sampling.innovation = True  # this will probably be superceded
  sampling.inverse_scaler = None

  model.beta_min = 0.01
  model.beta_max = 25.0

  solver.num_outer_steps = 1000
  solver.outer_solver = "EulerMaruyama"
  solver.inner_solver = None
  solver.m = 20  # Hutchinson estimator hyperparameter

  # optim
  config.seed = 2023

  return config
