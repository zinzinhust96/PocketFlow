import tensorflow as tf
from nets.abstract_model_helper import AbstractModelHelper
from utils.external.craft_tensorflow.model import craft as CRAFT
from utils.external.craft_tensorflow.loss import MSE_OHEM_Loss, calculate_fscore
from utils.external.craft_tensorflow.utils import decay_learning_rate
from datasets.synthtext_dataset import SynthTextDataset
import tensornets as nets


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('epoch', 1, ' ')
tf.app.flags.DEFINE_string('vgg_ckpt', '/hdd/Minhbq/PocketFlow_remove/backbone_models/vgg_16/vgg_16.ckpt', 'VGG tensorflow CKPT path')
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
    output = {'heatmaps': y_pred, 'is_train': tf.constant(is_train)}
    return output

class ModelHelper(AbstractModelHelper):
    """Model helper for creating a CRAFT model for the SynthText dataset."""

    def __init__(self, data_format='channels_last'):
        """Constructor function."""

        # class-independent initialization
        super(ModelHelper, self).__init__(data_format)

        # initialize training & evaluation subsets
        self.dataset_train = SynthTextDataset(is_train=True)
        self.dataset_eval = SynthTextDataset(is_train=False)

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
        * labels: dict
        * outputs: heatmaps
        # * data_format: data format ('channels_last' OR 'channels_first')

        Returns:
        * loss: loss calculated
        * metrics: dict (metrics that calculated (accuracy,...))
        """
        heatmaps = outputs['heatmaps']
        is_train = outputs['is_train']
        weight_characters = labels['weight_characters']
        weight_affinitys = labels['weight_affinitys']
        # char_coords = labels['char_coords']
        
        confident_maps = tf.ones_like(weight_characters)
        label = tf.stack((weight_characters, weight_affinitys), axis=-1)

        def true_fn():
            return MSE_OHEM_Loss(heatmaps, label, confident_maps, True)

        def false_fn():
            return MSE_OHEM_Loss(heatmaps, label, confident_maps, False)
            
        loss = tf.cond(is_train, true_fn, false_fn)
        metrics = {}
        # fscore = tf.py_function(calculate_fscore, [weight_characters, weight_affinitys, char_coords], (tf.float32))
        # metrics = {'fscore': fscore}
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

        return 'synthtext'