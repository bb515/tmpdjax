# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Training NCSN++ on Church with VE SDE."""

from configs.default_lsun_configs import get_default_configs


def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.sde = 'vesde'
  training.continuous = True

  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'reverse_diffusion'
  sampling.corrector = 'langevin'

  # data
  data = config.data
  data.dataset = 'FFHQ'
  data.image_size = 256
  # NOTE: It has to be a name of a tfrecords file
  # NOTE: The ffhq-rn record stands for 2**n image_size
  data.tfrecords_path = './assets/ffhq/ffhq-r08.tfrecords'

  # model
  model = config.model
  model.name = 'ncsnpp'
  model.sigma_max = 348
  model.scale_by_sigma = True
  model.ema_rate = 0.999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 1, 2, 2, 2, 2, 2)
  model.num_res_blocks = 2
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = True
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'output_skip'
  model.progressive_input = 'input_skip'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.
  model.fourier_scale = 16
  model.conv_size = 3

  # optim
  config.seed = 2023
  sampling.cs_method = 'TMPD2023bvjpplus'

  sampling.stack_samples = False
  sampling.noise_std = 0.01
  sampling.denoise = True
  sampling.inverse_scaler = None
  evaluate = config.eval
  evaluate.begin_ckpt = 48
  evaluate.end_ckpt = 48
  evaluate.batch_size = 10
  evaluate.pmap = False
  solver = config.solver
  # solver.num_outer_steps = config.model.num_scales
  solver.num_outer_steps = 100
  solver.outer_solver = 'eulermaruyama'
  # solver.outer_solver = 'DDIMVE'
  # solver.outer_solver = 'SMLD'
  solver.eta = 1.0  # DDIM hyperparameter

  # inpainting half
  # solver.dps_scale_hyperparameter = 1.0  # for noise_std=0.01
  # solver.dps_scale_hyperparameter = 1.0  # for noise_std=0.05
  # solver.dps_scale_hyperparameter = 0.5  # for noise_std=0.1

  # inpainting square
  # solver.dps_scale_hyperparameter = 1.0  # for noise_std=0.01
  # solver.dps_scale_hyperparameter = 1.0  # for noise_std=0.05
  solver.dps_scale_hyperparameter = 0.5 # for noise_std=0.1

  # superresolution 4bicubic
  # solver.dps_scale_hyperparameter = 1.0 # for noise_std=0.01
  # solver.dps_scale_hyperparameter = 1.0 # for noise_std=0.05
  # solver.dps_scale_hyperparameter = 0.5  # for noise_std=0.1

  # superresolution 8bicubic
  # solver.dps_scale_hyperparameter = 1.0 # for noise_std=0.01
  # solver.dps_scale_hyperparameter = 0.5 # for noise_std=0.05
  # solver.dps_scale_hyperparameter = 0.5  # for noise_std=0.1
  return config
