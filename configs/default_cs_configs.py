import ml_collections


def get_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 64
  training.n_iters = 2400001
  training.snapshot_freq = 50000
  training.log_epochs_freq = 10
  training.log_step_freq = 8
  training.eval_freq = 100
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 5000
  ## produce samples at each snapshot.
  training.snapshot_sampling = True
  training.likelihood_weighting = False
  training.score_scaling = True
  training.continuous = True
  training.n_jitted_steps = 1
  training.pmap = False
  training.reduce_mean = True
  training.pointwise_t = False

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.method = 'pc'
  sampling.predictor = 'reverse_diffusion'
  sampling.corrector = 'langevin'
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.206  # A hyperparameter of the corrector
  sampling.projection_sigma_rate = 1.586
  sampling.cs_solver = 'projection'
  sampling.expansion = 4
  sampling.coeff = 1.
  sampling.n_projections = 23
  sampling.task = 'ct'
  sampling.lambd = 0.5
  sampling.denoise_override = True
  sampling.stack_samples = False
  sampling.store_H = False

  # evaluation
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.ckpt_id = 101
  evaluate.batch_size = 128
  evaluate.enable_sampling = True
  evaluate.num_samples = 50000
  evaluate.enable_loss = True
  evaluate.enable_bpd = False
  evaluate.bpd_dataset = 'test'

  # data
  config.data = data = ml_collections.ConfigDict()
  data.random_flip = True
  data.uniform_dequantization = False
  data.centered = False
  data.num_channels = 1

  # model
  config.model = model = ml_collections.ConfigDict()
  model.name = 'mlp'
  model.sigma_max = 378.
  model.sigma_min = 0.01
  model.num_scales = 1000
  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.
  model.embedding_type = 'fourier'

  # solver
  config.solver = solver = ml_collections.ConfigDict()
  config.solver.num_outer_steps = 1000
  config.solver.num_inner_steps = 1
  config.solver.outer_solver = 'EulerMaruyama'
  config.solver.dt = None
  config.solver.epsilon = None
  config.solver.inner_solver = None
  config.solver.snr = None

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  config.seed = 42

  return config
