import os
import tensorflow as tf
from datasets.abstract_dataset import AbstractDataset
import sys
import numpy as np
# from base_models.craft.data_manipulation import resize
from utils.external.craft_tensorflow.icdar15_preprocessing import preprocess_image, preprocess_label
from scipy.io import loadmat
from tqdm import tqdm
from utils.external.craft_tensorflow.data_loader import Generator

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("nb_smpls_train", 99, "# of samples for training")
tf.app.flags.DEFINE_integer("nb_smpls_val", 45, "# of samples for validation")
tf.app.flags.DEFINE_integer("nb_smpls_eval", 45, "# of samples for evaluation")
tf.app.flags.DEFINE_integer("batch_size", 1, "batch size per GPU for training")
tf.app.flags.DEFINE_integer("batch_size_eval", 1, "batch size for evaluation")

# ILSVRC-12 specifications
IMAGE_SIDE = 768
IMAGE_CHN = 3

def parse_fn(image, label, is_train):
    """Parse image & labels from the serialized data.
    Args:
    * image: image tensor
    * label: label tensor
    * is_train: whether data augmentation should be applied
    Returns:
    * image: image tensor
    * label: (weight_character + weight_affinity) tensor
    """
    # label = {'weight_characters': weight_character, 'weight_affinitys': weight_affinity}

    return image, label

class SynthTextDataset(AbstractDataset):
    def __init__(self, is_train):
        # initialize the base class
        super(SynthTextDataset, self).__init__(is_train)
        # choose local files or HDFS files w.r.t. FLAGS.data_disk
        if FLAGS.data_disk == "local":
            assert (
                FLAGS.data_dir_local is not None
            ), "<FLAGS.data_dir_local> must not be None"
            data_dir = FLAGS.data_dir_local
        elif FLAGS.data_disk == "hdfs":
            assert (
                FLAGS.data_hdfs_host is not None and FLAGS.data_dir_hdfs is not None
            ), "both <FLAGS.data_hdfs_host> and <FLAGS.data_dir_hdfs> must not be None"
            data_dir = FLAGS.data_hdfs_host + FLAGS.data_dir_hdfs
        else:
            raise ValueError("unrecognized data disk: " + FLAGS.data_disk)

        # configure file patterns & function handlers

        if is_train:
            self.file_path = os.path.join(data_dir, "train")
            self.batch_size = FLAGS.batch_size
        else:
            self.file_path = os.path.join(data_dir, "val")
            self.batch_size = FLAGS.batch_size_eval
        
        self.images = np.array([])
        self.weight_characters = np.array([])
        self.weight_affinitys = np.array([]) 

        # data generator
        data_generator = Generator(
            file_mat=os.path.join(self.file_path, 'synth', 'bg.mat'), 
            prefix_path=os.path.join(self.file_path, 'synth', 'img'),
            len_queue=10,
            num_workers=8,
            batch_size = self.batch_size)
        data_generator.start()

        for iter in tqdm(range(data_generator.num_batches)):
            images, weight_characters, weight_affinitys = data_generator.get_batch()
            self.images = np.vstack([self.images, images]) if self.images.size else images
            self.weight_characters = np.vstack([self.weight_characters, weight_characters]) if self.weight_characters.size else weight_characters
            self.weight_affinitys = np.vstack([self.weight_affinitys, weight_affinitys]) if self.weight_affinitys.size else weight_affinitys
        print(self.images.shape)
        self.labels = np.stack((self.weight_characters, self.weight_affinitys), axis=-1)
        num_samples = len(self.images)
        self.images = self.images[:int(num_samples / self.batch_size) * self.batch_size]
        self.labels = self.labels[:int(num_samples / self.batch_size) * self.batch_size]
        self.parse_fn = lambda x, y: parse_fn(x, y, is_train)
        data_generator.kill()

    def build(self, enbl_trn_val_split=False, sess=None):
        """Build iterator(s) for tf.data.Dataset() object.

        Args:
        * enbl_trn_val_split: whether to split into training & validation subsets

        Returns:
        * iterator_trn: iterator for the training subset
        * iterator_val: iterator for the validation subset
        OR
        * iterator: iterator for the chosen subset (training OR testing)
        """

        # create a tf.data.Dataset() object from NumPy arrays
        self.images_placeholder = tf.placeholder(self.images.dtype, self.images.shape)
        self.labels_placeholder = tf.placeholder(self.labels.dtype, self.labels.shape)
        # print('>>>>> self.images_placeholder', self.images_placeholder)
        # print('>>>>> self.labels_placeholder', self.labels_placeholder)

        dataset = tf.data.Dataset.from_tensor_slices((self.images_placeholder, self.labels_placeholder))
        dataset = dataset.map(self.parse_fn, num_parallel_calls=FLAGS.nb_threads)

        data = (self.images, self.labels)
        # create iterators for training & validation subsets separately
        if self.is_train and enbl_trn_val_split:
            iterator_val = self.__make_iterator(dataset.take(FLAGS.nb_smpls_val), data=data, sess = sess)
            iterator_trn = self.__make_iterator(dataset.skip(FLAGS.nb_smpls_val), data=data, sess = sess)
            return iterator_trn, iterator_val

        return self.__make_iterator(dataset, data=data, sess = sess), data

    def __make_iterator(self, dataset, data, sess = None):
        """Make an iterator from tf.data.Dataset.

        Args:
        * dataset: tf.data.Dataset object

        Returns:
        * iterator: iterator for the dataset
        """
        images, labels = data

        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=FLAGS.buffer_size))
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(FLAGS.prefetch_size)
        iterator = dataset.make_initializable_iterator()

        # print('iterator >>>>>>>>>>>>>>>.', iterator.initializer)

        # feed the placeholder with data
        sess.run(iterator.initializer, feed_dict={ self.images_placeholder: images, self.labels_placeholder: labels }) 

        return iterator