import tensorflow as tf
from nets.abstract_model_helper import AbstractModelHelper
from utils.external.craft_tensorflow.model import craft as CRAFT
from utils.external.craft_tensorflow.loss import MSE_OHEM_Loss
from utils.external.craft_tensorflow.utils import decay_learning_rate, calculate_fscore
from datasets.icdar15_dataset import ICDAR15Dataset

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('epoch', 20, ' ')
# tf.app.flags.DEFINE_integer('batch_size', 20, ' ')
tf.app.flags.DEFINE_float('learning_rate', 0.001, ' ')
tf.app.flags.DEFINE_float('decay_factor_lr', 0.0125/4000, ' ')
tf.app.flags.DEFINE_float('momentum', 0.9, 'momentum coefficient')

def forward_fn(inputs, is_train, data_format):
    """Forward pass function.

    Args:
    * inputs: inputs to the network's forward pass
    * is_train: whether to use the forward pass with training operations inserted
    * data_format: data format ('channels_last' OR 'channels_first')

    Returns:
    * outputs from the network's forward pass
    """
    VGG16, y_pred = CRAFT(inputs, is_train)
    return y_pred

class ModelHelper(AbstractModelHelper):
    """Model helper for creating a CRAFT model for the ICDAR-15 dataset."""

    def __init__(self, data_format='channels_last'):
        """Constructor function."""

        # class-independent initialization
        super(ModelHelper, self).__init__(data_format)

        # initialize training & evaluation subsets
        self.dataset_train = ICDAR15Dataset(is_train=True)
        self.dataset_eval = ICDAR15Dataset(is_train=False)

    def build_dataset_train(self, enbl_trn_val_split=False):
        """Build the data subset for training, usually with data augmentation."""

        return self.dataset_train.build(enbl_trn_val_split)

    def build_dataset_eval(self):
        """Build the data subset for evaluation, usually without data augmentation."""

        return self.dataset_eval.build()

    def forward_train(self, inputs):
        """Forward computation at training."""

        return forward_fn(inputs, is_train=True, data_format=self.data_format)

    def forward_eval(self, inputs):
        """Forward computation at evaluation."""

        return forward_fn(inputs, is_train=False, data_format=self.data_format)

    def calc_loss(self, labels, outputs, trainable_vars):
        """Calculate loss (and some extra evaluation metrics).
        Args:
        * inputs: inputs to the network's forward pass
        * is_train: whether to use the forward pass with training operations inserted
        * data_format: data format ('channels_last' OR 'channels_first')

        Returns:
        * loss: loss calculated
        * metrics: dict (metrics that calculated (accuracy,...))
        """
        
        # conf_map = labels['conf_map']
        # y = labels['y']
        # loss = MSE_OHEM_Loss(outputs, y, conf_map)
        loss = tf.constant(0.5)
        # metrics = {'fscore': calculate_fscore(outputs, y)}
        metrics = {'acc' : 0.9}
        return loss, metrics

    def setup_lrn_rate(self, global_step):
        """Setup the learning rate (and number of training iterations)."""
        lr = FLAGS.learning_rate * 1.0 / (1.0 + FLAGS.decay_factor_lr*tf.cast(global_step, tf.float32))
        nb_iter = FLAGS.nb_smpls_train*FLAGS.epoch//FLAGS.batch_size
        return lr, nb_iter

    @property
    def model_name(self):
        """Model's name."""

        return 'craft'

    @property
    def dataset_name(self):
        """Dataset's name."""

        return 'icdar15'