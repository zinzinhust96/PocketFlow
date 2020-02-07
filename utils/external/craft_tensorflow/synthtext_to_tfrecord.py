import os
import tensorflow as tf
from icdar15 import load_icdar15_label
import cv2
import numpy as np
from scipy.io import loadmat
import config as CONFIG
import six
import base64
from augment import rand_augment
from data_manipulation import resize, generate_target, generate_affinity, normalize_mean_variance
os.environ["CUDA_VISIBLE_DEVICES"]="1"

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

def scale(img, newrange=(0,255), eps = 1e-8):
    min_ = np.min(img)
    max_ = np.max(img)
    img = (newrange[1] - newrange[0])*(img - min_)/(max_ - min_ + eps) + newrange[0]
    return img

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
    # print(char_bbs.reshape(-1, 4, 2)[:10])
    char_bbs = char_bbs.reshape(-1, 8)
    for ind in range(char_bbs.shape[0]):
        obj = char_bbs[ind].flatten()
        obj = [int(number) for number in obj]
        [l.append(point) for l, point in zip([x_top_left, y_top_left,
                                              x_top_right, y_top_right,
                                              x_bottom_right, y_bottom_right,
                                              x_bottom_left, y_bottom_left], obj)]
    # print(obj)
    # print('{}, {}, {}, {}, {}, {}, {}, {}'.format(x_top_left, y_top_left, x_top_right, y_top_right,
    #                                           x_bottom_right, y_bottom_right,
    #                                           x_bottom_left, y_bottom_left))
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
                "image/image/encoded": _float_feature(image_buffer[0]),
                "image/weight_character/encoded": _float_feature(image_buffer[1]),
                "image/weight_affinity/encoded": _float_feature(image_buffer[2]),
            }
        )
    )
    return example

def encode_jpeg(data):
    format = 'rgb'
    if len(data.shape) == 2:
        format = 'grayscale'
        data = data[..., None]
    print(data)
    g = tf.Graph()
    with g.as_default():
        data_t = tf.placeholder(tf.uint8)
        op = tf.image.encode_jpeg(data_t, format=format, quality=100)
        init = tf.initialize_all_variables()

    with tf.Session(graph=g) as sess:
        sess.run(init)
        data_np = sess.run(op, feed_dict={ data_t: data })
    return data_np

if __name__ == "__main__":
    SYNTH_IMG_PATH = '/hdd/UBD/background-images/data_generated/Pocketflow/val/synth'
    OUTPUT_PATH = '/hdd/Minhbq/syntext_data/val.tfrecord'
    AUGUMENT = True
    writer = tf.python_io.TFRecordWriter(OUTPUT_PATH)
    mat = loadmat(os.path.join(SYNTH_IMG_PATH, 'bg.mat'))
    imnames = mat['imnames'][0] # ['file1.png', 'file2.png', ..., 'finen.png']
    charBBs = mat['charBB'][0]  # number of images, 2, 4, num_character
    txts = mat['txt'][0]
    for i, imname in enumerate(imnames):
        filename = imname[0]
        print('{}/{}: Processing {}'.format(i, len(imnames), filename), end='\r')
        img_path = os.path.join(SYNTH_IMG_PATH, 'img', filename)
        # Read RGB
        image = cv2.imread(img_path)[:,:,::-1]
        height, width, channels = image.shape
        # Encode image into buffers
        # character bounding box
        char_bbs = charBBs[i] #shape: (2, 4, ?)
        # character text
        txt = txts[i]
        word_texts = txt.copy()
        # word_texts example: ['clusiaceae', '3HC', 'achimenes', '228X6', 'OBZG0', 'ANTIDOTES', 'MUTEST', 'カルラ']
        word_texts = [word_text.strip() for word_text in word_texts]

        # Get weight_character and weight_affinity
        image, character = resize(image, char_bbs.copy())
        chw_image = (image.copy()).transpose(2, 0, 1)
        weight_character = generate_target(chw_image.shape, character.copy())
        weight_affinity, affinity_bbox = generate_affinity(chw_image.shape, character.copy(), word_texts.copy())
        if AUGUMENT:
            imgs = rand_augment(list([image.copy(), weight_character.copy(), weight_affinity.copy()]))
        else:
            imgs = [image.copy(), weight_character.copy(), weight_affinity.copy()]
        imgs[0] = normalize_mean_variance(imgs[0][:,:,::-1])
        # print(np.unique(imgs[0]))
        # print(np.unique(imgs[1]))
        # print(np.max(imgs[2]))
        image_data = [img.flatten().tolist() for img in imgs]
        example = convert_to_example(img_path, filename, image_data, char_bbs, word_texts,
                                height, width, channels)
        # print(image_data[0].shape)
        # print(image_data[1].shape)
        # print(image_data[2].shape)

        # break
        writer.write(example.SerializeToString())
    writer.close()