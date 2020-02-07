import os
import tensorflow as tf
from icdar15 import load_icdar15_label
import cv2
import numpy as np
from scipy.io import loadmat
import config as CONFIG
import six
import base64

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_list_feature(value):
    """Wrapper for inserting a list of bytes features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    if isinstance(value, six.string_types):
        value = six.binary_type(value, encoding="utf-8")
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to_example(
    filename, image_name, image_buffer, char_bbs, word_texts, height, width, channel):
    """Build an Example proto for an example.

        Args:
        filename: string, path to an image file, e.g., '/path/to/example.JPG'
        image_buffer: string, JPEG encoding of RGB image
        char_bbs: List of corner coordinates for each character
        word_texts: List of corresponding word value of character
        height: integer, image height in pixels
        width: integer, image width in pixels
        Returns:
        Example proto
    """
    x_top_left = []
    y_top_left = []
    x_top_right = []
    y_top_right = []
    x_bottom_right = []
    y_bottom_right = []
    x_bottom_left = []
    y_bottom_left = []
    word_texts = [word.encode('utf8') for word in word_texts]
    for ind in range(char_bbs.shape[0]):
        obj = char_bbs[ind].flatten()
        obj = [int(number) for number in obj]
        [l.append(point) for l, point in zip([x_top_left, y_top_left,
                                              x_top_right, y_top_right,
                                              x_bottom_right, y_bottom_right,
                                              x_bottom_left, y_bottom_left], obj)]
    image_format = "JPG"
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/height": _int64_feature(height),
                "image/width": _int64_feature(width),
                "image/channels": _int64_feature(channels),
                "image/shape": _int64_feature([height, width, channels]),
                "image/object/x_top_left": _float_feature(x_top_left),
                "image/object/y_top_left": _float_feature(y_top_left),
                "image/object/x_top_right": _float_feature(x_top_right),
                "image/object/y_top_right": _float_feature(y_top_right),
                "image/object/x_bottom_right": _float_feature(x_bottom_right),
                "image/object/y_bottom_right": _float_feature(y_bottom_right),
                "image/object/x_bottom_left": _float_feature(x_bottom_left),
                "image/object/y_bottom_left": _float_feature(y_bottom_left),
                "image/object/words": _bytes_list_feature(word_texts),
                "image/format": _bytes_feature(image_format),
                "image/filename": _bytes_feature(image_name.encode("utf8")),
                "image/encoded": _bytes_feature(image_buffer),
            }
        )
    )
    return example

def encode_jpeg(data):
    g = tf.Graph()
    with g.as_default():
        data_t = tf.placeholder(tf.uint8)
        op = tf.image.encode_jpeg(data_t, format='rgb', quality=100)
        init = tf.initialize_all_variables()

    with tf.Session(graph=g) as sess:
        sess.run(init)
        data_np = sess.run(op, feed_dict={ data_t: data })
    return data_np

if __name__ == "__main__":
    SYNTH_IMG_PATH = '/hdd/UBD/background-images/data_generated/Pocketflow/val/synth'
    OUTPUT_PATH = '/hdd/namdng/tf_record_dataset/synthtext/val.tfrecord'
    writer = tf.python_io.TFRecordWriter(OUTPUT_PATH)
    mat = loadmat(os.path.join(SYNTH_IMG_PATH, 'bg.mat'))
    imnames = mat['imnames'][0] # ['file1.png', 'file2.png', ..., 'finen.png']
    charBBs = mat['charBB'][0]  # number of images, 2, 4, num_character
    txts = mat['txt'][0]
    for i, imname in enumerate(imnames):
        filename = imname[0]
        print('{}/{}: Processing {}'.format(i, len(imnames), filename), end='\r')
        img_path = os.path.join(SYNTH_IMG_PATH, 'img', filename)
        img = cv2.imread(img_path)[:,:,::-1]
        height, width, channels = img.shape
        image_data = encode_jpeg(img)
        # character bounding box
        char_bbs = charBBs[i]
        char_bbs = np.transpose(char_bbs, (2, 1, 0))
        # character text
        txt = txts[i]
        word_texts = txt.copy()
        print(word_texts)
        example = convert_to_example(img_path, filename, image_data, char_bbs, word_texts,
                                height, width, channels)

        writer.write(example.SerializeToString())
writer.close()