import tensorflow as tf
from nets.abstract_model_helper import AbstractModelHelper
from utils.external.craft_tensorflow.model import craft as CRAFT
from utils.external.craft_tensorflow.loss import MSE_OHEM_Loss, calculate_fscore
from utils.external.craft_tensorflow.utils import decay_learning_rate
from datasets.synthtext_dataset import SynthTextDataset
import tensornets as nets


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('epoch', 10, ' ')
tf.app.flags.DEFINE_string('vgg_ckpt', '/hdd/Minhbq/Deep-Compression/pretrained/vgg16.ckpt', 'VGG tensorflow CKPT path')
tf.app.flags.DEFINE_boolean('is_resume', False, 'Whether to load vgg pretrained model')
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
    y_pred = CRAFT(inputs, is_train)
    # output = {'heatmaps': y_pred, 'is_train': tf.constant(is_train)}
    model_scope = tf.get_default_graph().get_name_scope()
    
    return y_pred, model_scope

class ModelHelper(AbstractModelHelper):
    """Model helper for creating a CRAFT model for the SynthText dataset."""

    def __init__(self, data_format='channels_last'):
        """Constructor function."""

        # class-independent initialization
        super(ModelHelper, self).__init__(data_format)

        # initialize training & evaluation subsets
        self.dataset_train = SynthTextDataset(is_train=True)
        self.dataset_eval = SynthTextDataset(is_train=False)
        self.model_scope = None
        self.trainable_vars = None

    def build_dataset_train(self, enbl_trn_val_split=False):
        """Build the data subset for training, usually with data augmentation."""

        return self.dataset_train.build(enbl_trn_val_split)

    def build_dataset_eval(self):
        """Build the data subset for evaluation, usually without data augmentation."""

        return self.dataset_eval.build()

    def forward_train(self, inputs):
        """Forward computation at training."""
        outputs, self.model_scope = forward_fn(inputs, is_train=True, data_format=self.data_format)
        self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_scope)
        return outputs

    def forward_eval(self, inputs):
        """Forward computation at evaluation."""
        outputs, self.model_scope = forward_fn(inputs, is_train=False, data_format=self.data_format)
        self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_scope)
        return outputs

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

        # weight_characters = labels['weight_characters']
        # weight_affinitys = labels['weight_affinitys']
        # char_coords = labels['char_coords']
        confident_maps = tf.ones_like(labels[:, :, :, 0])
        # label = tf.stack((weight_characters, weight_affinitys), axis=-1)
        loss = MSE_OHEM_Loss(outputs, labels, confident_maps)
            
        # loss = tf.cond(is_train, true_fn, false_fn)
        # metrics = {}
        # fscore = tf.py_function(calculate_fscore, [weight_characters, weight_affinitys, char_coords], (tf.float32))
        metrics = {'accuracy': -loss}
        return loss, metrics

    def warm_start(self, sess):
        """Initialize the model for warm-start.

        Args:
        * sess: TensorFlow session
        """
        if FLAGS.is_resume:
            ckpt_path = FLAGS.vgg_ckpt
            tf.logging.info('restoring model weights from ' + ckpt_path)
            reader = tf.train.NewCheckpointReader(ckpt_path)
            vars_list_avail = {}
            for var in self.trainable_vars:
                tensor_name = var.name[var.name.find('VGG'):-2].strip()
                if tensor_name != '' and reader.has_tensor(tensor_name):
                    print(tensor_name)
                    # tensor_name = var.name
                    vars_list_avail[tensor_name] = var
            # print(vars_list_avail)
            saver = tf.train.Saver(vars_list_avail, reshape=False)
            saver.build()
            saver.restore(sess, ckpt_path)


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