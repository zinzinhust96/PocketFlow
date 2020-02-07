import tensorflow as tf
from nets.abstract_model_helper import AbstractModelHelper
from utils.external.craft_tensorflow.model import craft as CRAFT
from utils.external.craft_tensorflow.loss import MSE_OHEM_Loss
from utils.external.craft_tensorflow.utils import decay_learning_rate, calculate_fscore
from datasets.icdar15_dataset import ICDAR15Dataset
from utils.external.craft_tensorflow.icdar15_preprocessing import create_weakly_batch, gen_batch_image_contain_only_words, normalize_mean_variance, create_label

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('epoch', 20, ' ')
# tf.app.flags.DEFINE_integer('batch_size', 20, ' ')
tf.app.flags.DEFINE_float('learning_rate', 0.001, ' ')
tf.app.flags.DEFINE_float('decay_factor_lr', 0.0125/4000, ' ')
tf.app.flags.DEFINE_float('momentum', 0.9, 'momentum coefficient')

def forward_fn(inputs, labels, is_train, data_format):
    """Forward pass function.

    Args:
    * inputs: inputs to the network's forward pass
    * is_train: whether to use the forward pass with training operations inserted
    * data_format: data format ('channels_last' OR 'channels_first')

    Returns:
    * outputs from the network's forward pass
    """
    word_coords = labels['word_coords']
    word_texts = labels['word_texts']
    weakly_batch, lookup_heatmaps = gen_batch_image_contain_only_words(inputs, word_coords)
    # print(weakly_batch)
    _, heatmaps = CRAFT(weakly_batch, False)
    images, weight_characters, weight_affinitys, confident_maps = tf.py_function(create_label, 
                                                                [inputs, weakly_batch, heatmaps, lookup_heatmaps], 
                                                                (tf.float32, tf.float32, tf.float32, tf.float32))
    images.set_shape(inputs.get_shape())
    VGG16, y_pred = CRAFT(images, is_train)
    outputs = {'y_pred': y_pred, 'weight_characters': weight_characters, 
                'weight_affinitys': weight_affinitys, 'confident_maps' : confident_maps}

    return outputs

def forward_eval(inputs, is_train, data_format):
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
        super(ModelHelper, self).__init__(data_format, forward_w_labels=True)

        # initialize training & evaluation subsets
        self.dataset_train = ICDAR15Dataset(is_train=True)
        self.dataset_eval = ICDAR15Dataset(is_train=False)

    def build_dataset_train(self, enbl_trn_val_split=False):
        """Build the data subset for training, usually with data augmentation."""

        return self.dataset_train.build(enbl_trn_val_split)

    def build_dataset_eval(self):
        """Build the data subset for evaluation, usually without data augmentation."""

        return self.dataset_eval.build()

    def forward_train(self, inputs, labels):
        """Forward computation at training."""

        return forward_fn(inputs, labels, is_train=True, data_format=self.data_format)

    def forward_eval(self, inputs, labels):
        """Forward computation at evaluation."""

        return forward_fn(inputs, labels, is_train=False, data_format=self.data_format)

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
        y_pred = outputs['y_pred']
        weight_characters = outputs['weight_characters']
        weight_affinitys = outputs['weight_affinitys']
        confident_maps = outputs['confident_maps']
        y = tf.stack([weight_characters, weight_affinitys], axis=-1)
        print(y)
        loss = MSE_OHEM_Loss(y_pred, y, confident_maps)
        # metrics = {'fscore': calculate_fscore(y_pred, y)}
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