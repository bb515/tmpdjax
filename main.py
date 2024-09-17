"""Image restoration experiments."""
import run_lib
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import tensorflow as tf
import os

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum("mode", None, ["eval_from_file",
                                 "eval_inpainting", "eval_super_resolution",
                                 "dps_search_inpainting", "dps_search_super_resolution",
                                 "sample",
                                 "inpainting", "super_resolution", "deblur", "jpeg"],
                  "Running mode: sample, inpainting, super_resolution or deblur")
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")
flags.DEFINE_float("noise_std", 0.0, "noise standard")
flags.mark_flags_as_required(["workdir", "config", "mode"])


def main(argv):
    tf.config.experimental.set_visible_devices([], "GPU")
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'  # Less prone to GPU memory fragmentation, which should prevent OOM on CIFAR10
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.93'  # preallocate 93 percent of memory, which may cause OOM when the JAX program starts

    FLAGS.config.sampling.noise_std = FLAGS.noise_std
    if FLAGS.mode == "sample":
        run_lib.sample(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)
    elif FLAGS.mode == "eval_from_file":
        # run_lib.evaluate_from_file(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)
        run_lib.revaluate_from_file(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)
    elif FLAGS.mode == "eval_inpainting":
        run_lib.evaluate_inpainting(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)
    elif FLAGS.mode == "eval_super_resolution":
        run_lib.evaluate_super_resolution(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)
    elif FLAGS.mode == "dps_search_inpainting":
        run_lib.dps_search_inpainting(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)
    elif FLAGS.mode == "dps_search_super_resolution":
        run_lib.dps_search_super_resolution(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)
    elif FLAGS.mode == "inpainting":
        run_lib.inpainting(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)
    elif FLAGS.mode == "super_resolution":
        run_lib.super_resolution(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)
    elif FLAGS.mode == "deblur":
        run_lib.deblur(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)
    elif FLAGS.mode == "jpeg":
        run_lib.jpeg(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)


if __name__ == "__main__":
    app.run(main)
