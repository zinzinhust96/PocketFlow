import os
import tensorflow as tf
from datasets.abstract_dataset import AbstractDataset
import sys
# from base_models.craft.data_manipulation import resize
from utils.external.craft_tensorflow.icdar15_preprocessing import preprocess_image, preprocess_label

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("nb_smpls_train", 1281167, "# of samples for training")
tf.app.flags.DEFINE_integer("nb_smpls_val", 10000, "# of samples for validation")
tf.app.flags.DEFINE_integer("nb_smpls_eval", 50000, "# of samples for evaluation")
tf.app.flags.DEFINE_integer("batch_size", 64, "batch size per GPU for training")
tf.app.flags.DEFINE_integer("batch_size_eval", 1, "batch size for evaluation")

# ILSVRC-12 specifications
IMAGE_SIDE = 768
IMAGE_CHN = 3

def parse_example_proto(example_serialized):
    """Parse image buffer, label, and bounding box from the serialized data.

    Args:
    * example_serialized: serialized example data

    Returns:
    * image_buffer: image buffer label
    * label: label tensor (not one-hot)
    * bbox: bounding box tensor
    """
    feature_map = {
        "image/encoded": tf.FixedLenFeature([], dtype=tf.string, default_value=""),
        "image/object/x_top_left": tf.VarLenFeature(dtype=tf.float32),
        "image/object/y_top_left": tf.VarLenFeature(dtype=tf.float32),
        "image/object/x_top_right": tf.VarLenFeature(dtype=tf.float32),
        "image/object/y_top_right": tf.VarLenFeature(dtype=tf.float32),
        "image/object/x_bottom_right": tf.VarLenFeature(dtype=tf.float32),
        "image/object/y_bottom_right": tf.VarLenFeature(dtype=tf.float32),
        "image/object/x_bottom_left": tf.VarLenFeature(dtype=tf.float32),
        "image/object/y_bottom_left": tf.VarLenFeature(dtype=tf.float32),
        "image/height": tf.FixedLenFeature([], dtype=tf.int64, default_value=0),
        "image/width": tf.FixedLenFeature([], dtype=tf.int64, default_value=0),
        "image/channels": tf.FixedLenFeature([], dtype=tf.int64, default_value=3)
        # "image/object/texts": tf.VarLenFeature([], dtype=tf.string)
    }

    features = tf.parse_single_example(example_serialized, feature_map)
    height = tf.cast(features['image/height'], dtype=tf.int32)
    width = tf.cast(features['image/width'], dtype=tf.int32)
    channel = tf.cast(features['image/channels'], dtype=tf.int32)
    shape = [width, height, channel]
    x_top_left = tf.expand_dims(features['image/object/x_top_left'].values, 0)
    y_top_left = tf.expand_dims(features['image/object/y_top_left'].values, 0)
    x_top_right = tf.expand_dims(features['image/object/x_top_right'].values, 0)
    y_top_right = tf.expand_dims(features['image/object/y_top_right'].values, 0)
    x_bottom_right = tf.expand_dims(features['image/object/x_bottom_right'].values, 0)
    y_bottom_right = tf.expand_dims(features['image/object/y_bottom_right'].values, 0)
    x_bottom_left = tf.expand_dims(features['image/object/x_bottom_left'].values, 0)
    y_bottom_left = tf.expand_dims(features['image/object/y_bottom_left'].values, 0)
    # word_texts =  tf.expand_dims(features['image/object/texts'].values, 0)

    word_coords = tf.reshape(tf.stack([tf.squeeze(x_top_left), tf.squeeze(y_top_left), 
               tf.squeeze(x_top_right), tf.squeeze(y_top_right),
               tf.squeeze(x_bottom_right), tf.squeeze(y_bottom_right),
               tf.squeeze(x_bottom_left), tf.squeeze(y_bottom_left)
              ], axis=1), (-1, 4, 2))
    # tl = tf.stack([tf.squeeze(x_top_left), tf.squeeze(y_top_left)], axis=1)
    # tr = tf.stack([tf.squeeze(x_top_right), tf.squeeze(y_top_right)], axis=1)
    # br = tf.stack([tf.squeeze(x_bottom_right), tf.squeeze(y_bottom_right)], axis=1)
    # bl = tf.stack([tf.squeeze(x_bottom_left), tf.squeeze(y_bottom_left)], axis=1)
    # word_coords = tf.concat([tl, tr, br, bl], axis=1)
    tf.print(word_coords, output_stream=sys.stdout)

    return features['image/encoded'], word_coords, shape




def parse_fn(example_serialized, is_train):
    """Parse image & labels from the serialized data.
    Args:
    * example_serialized: serialized example data
    * is_train: whether data augmentation should be applied

    Returns:
    * image: image tensor
    * word_coords: 
    """

    image_buffer, word_coords, shape = parse_example_proto(example_serialized)

    ######## decode and resize ########


    ######## decode and resize ########
    image = preprocess_image(image_buffer=image_buffer, output_side=IMAGE_SIDE, num_channels=IMAGE_CHN, is_training=is_train)
    ######## decode and resize ########
    word_coords = preprocess_label(input_shape = shape, character = tf.transpose(word_coords), side = IMAGE_SIDE)

    
    #return image and label used in learner to calculate loss
    # return image [None, 768, 768, 3], label [None, 768, 768, 2] + confident_map
    return image, word_coords


class ICDAR15Dataset(AbstractDataset):
    def __init__(self, is_train):
        # initialize the base class
        super(ICDAR15Dataset, self).__init__(is_train)
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
            self.file_pattern = os.path.join(data_dir, "*train*")
            self.batch_size = FLAGS.batch_size
        else:
            self.file_pattern = os.path.join(data_dir, "*val*")
            self.batch_size = FLAGS.batch_size_eval
        self.dataset_fn = tf.data.TFRecordDataset
        self.parse_fn = lambda x: parse_fn(x, is_train=is_train)
