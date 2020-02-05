import os
import tensorflow as tf
from icdar15 import load_icdar15_label
import cv2
import numpy as np
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


def load_icdar15_label(label_path):
    with open(label_path, mode='r', encoding='utf-8-sig') as f_r:
        file_content = f_r.readlines()

    word_boxes = []
    word_text = []
    valid_data = False

    for line in file_content:
        line = str(line)
        elements = line.split(',')

        if len(elements) != 9:
            word = ','.join(elements[8:]).rstrip()
            corner_coordinates = np.array(elements[:8])
            print('Special Text', label_path, elements, word)
        else:
            corner_coordinates = np.array(elements[:-1])
            word = elements[-1].rstrip()

        word_boxes.append(corner_coordinates)
        word_text.append(word)
            
        if word not in CONFIG.DONTCARE_LABEL:
            valid_data = True

    if valid_data:
        return word_boxes, word_text
    else:
        return [], []


def convert_to_example(
    filename, image_name, image_buffer, word_boxes, word_texts, height, width, channel):
    """Build an Example proto for an example.

        Args:
        filename: string, path to an image file, e.g., '/path/to/example.JPG'
        image_buffer: string, JPEG encoding of RGB image
        word_boxes: List of corner coordinates for each word
        word_texts: List of labels for bounding box
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
    for obj in word_boxes:
        obj = [int(number) for number in obj]
        [l.append(point) for l, point in zip([x_top_left, y_top_left,
                                              x_top_right, y_top_right,
                                              x_bottom_right, y_bottom_right,
                                              x_bottom_left, y_bottom_left], obj)]
    # print(x_top_left, y_top_left)
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
                "image/object/texts": _bytes_list_feature(word_texts),
                "image/format": _bytes_feature(image_format),
                "image/filename": _bytes_feature(image_name.encode("utf8")),
                "image/encoded": _bytes_feature(image_buffer),
            }
        )
    )
    return example


# DIRECTORY = '/home/thanhtm/Desktop/icdar2015/'
TRAIN_LIST = '/home/thanhtm/Desktop/icdar2015/train_images/'
# TEST_LIST = '/home/thanhtm/Desktop/icdar2015/test_images'
LABEL_DIRECTORY = '/home/thanhtm/Desktop/icdar2015/train_gts/'
OUTPUT_PATH = '/hdd/Minhbq/TF-record-datasets/craft/train.tfrecord'
writer = tf.python_io.TFRecordWriter(OUTPUT_PATH)
filenames = os.listdir(TRAIN_LIST)
for i, filename in enumerate(filenames):
    print('{}/{}: Processing {}'.format(i, len(filenames), filename), end='\r')
    img_path = TRAIN_LIST + filename
    label_path = LABEL_DIRECTORY + filename + '.txt'
    img = cv2.imread(img_path)[:,:,::-1]
    height, width, channels = img.shape
    word_boxes, word_text = load_icdar15_label(label_path)
    print(word_boxes.shape)
    _, img_buffer = cv2.imencode('.jpg', img)
    img_buffer = base64.b64encode(img_buffer)    
    example = convert_to_example(img_path, filename, img_buffer, word_boxes, word_text, 
                                height, width, channels)
    break    
    # writer.write(example.SerializeToString())
# writer.close()