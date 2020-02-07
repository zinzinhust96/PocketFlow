"""Execution script for CRAFT models on the SynthText dataset."""

import traceback
import tensorflow as tf

from nets.craft_at_synthtext import ModelHelper
from learners.learner_utils import create_learner

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("log_dir", "./logs", "logging directory")
tf.app.flags.DEFINE_boolean("enbl_multi_gpu", False, "enable multi-GPU training")
tf.app.flags.DEFINE_string("learner", "full-prec", "learner's name")
tf.app.flags.DEFINE_string("exec_mode", "train", "execution mode: train / eval")
tf.app.flags.DEFINE_boolean("debug", False, "debugging information")

def main(unused_argv):
  """Main entry."""
  try:
    # setup the TF logging routine
    if FLAGS.debug:
      tf.logging.set_verbosity(tf.logging.DEBUG)
    else:
      tf.logging.set_verbosity(tf.logging.INFO)
    sm_writer = tf.summary.FileWriter(FLAGS.log_dir)
    # display FLAGS's values
    tf.logging.info("FLAGS:")
    for key, value in FLAGS.flag_values_dict().items():
      tf.logging.info("{}: {}".format(key, value))

    # build the model helper & learner
    model_helper = ModelHelper()
    learner = create_learner(sm_writer, model_helper)
    # execute the learner
    if FLAGS.exec_mode == "train":
      learner.train()
    elif FLAGS.exec_mode == "eval":
      learner.download_model()
      learner.evaluate()
    else:
      raise ValueError("unrecognized execution mode: " + FLAGS.exec_mode)
  except ValueError:
    traceback.print_exc()
    return 1  # exit with errors

if __name__ == '__main__':
  tf.app.run()
