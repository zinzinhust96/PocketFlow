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
import multiprocessing
import threading
from datetime import datetime
os.environ["CUDA_VISIBLE_DEVICES"]="0"

DATA_MODE = 'train'
SYNTH_IMG_PATH = '/hdd/UBD/background-images/data_generated/Pocketflow/{}/synth'.format(DATA_MODE)
NUMBER_OF_THREAD = 1
AUGUMENT = True if DATA_MODE == 'train' else False
print('AUGMENT: ', AUGUMENT)

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

class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that converts CMYK JPEG data to RGB JPEG data.
        self._cmyk_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
        self._cmyk_to_rgb = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                            feed_dict={self._png_data: image_data})

    def cmyk_to_rgb(self, image_data):
        return self._sess.run(self._cmyk_to_rgb,
                            feed_dict={self._cmyk_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                            feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

def _process_image(filename, coder):
    """Process a single image file.
    Args:
        filename: string, path to an image file e.g., '/path/to/example.JPG'.
        coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
        image_buffer: string, JPEG encoding of RGB image.
        height: integer, image height in pixels.
        width: integer, image width in pixels.
    """
    # Read the image file.
    image_data = tf.gfile.GFile(filename, 'rb').read()

    print('Converting PNG to JPEG for %s' % filename)
    image_data = coder.png_to_jpeg(image_data)

    print('Encoded')

    # Decode the RGB JPEG.
    image = coder.decode_jpeg(image_data)
    print('Decoded')

    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]
    assert channels == 3

    return image_data, image, height, width, channels

def convert_to_example(
    filename, image_name, image_buffer, weight_character_buffer, weight_affinity_buffer, char_bbs, word_texts, height, width, channels):
    # print(image_buffer[0].shape, image_buffer[1].shape, image_buffer[2].shape)
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
    char_bbs = char_bbs.reshape(-1, 8)
    for ind in range(char_bbs.shape[0]):
        obj = char_bbs[ind]
        # obj = [int(number) for number in obj]
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
                # "image/height": _int64_feature(height),
                # "image/width": _int64_feature(width),
                # "image/channels": _int64_feature(channels),
                # "image/shape": _int64_feature([height, width, channels]),
                "image/object/x_top_left": _float_feature(x_top_left),
                "image/object/y_top_left": _float_feature(y_top_left),
                "image/object/x_top_right": _float_feature(x_top_right),
                "image/object/y_top_right": _float_feature(y_top_right),
                "image/object/x_bottom_right": _float_feature(x_bottom_right),
                "image/object/y_bottom_right": _float_feature(y_bottom_right),
                "image/object/x_bottom_left": _float_feature(x_bottom_left),
                "image/object/y_bottom_left": _float_feature(y_bottom_left),
                # "image/object/words": _bytes_list_feature(word_texts),
                "image/format": _bytes_feature(image_format),
                "image/filename": _bytes_feature(image_name.encode("utf8")),
                "image/image/encoded": _bytes_feature(image_buffer),
                "image/weight_character/encoded": _bytes_feature(weight_character_buffer),
                "image/weight_affinity/encoded": _bytes_feature(weight_affinity_buffer),
            }
        )
    )
    return example

def encode_jpeg(data):
    format_type = 'rgb'
    if len(data.shape) <= 2:
        data = data[..., None]
        format_type = 'grayscale'
    g = tf.Graph()
    with g.as_default():
        data_t = tf.placeholder(tf.uint8)
        op = tf.image.encode_jpeg(data_t, format=format_type, quality=100)
        init = tf.initialize_all_variables()

    with tf.Session(graph=g) as sess:
        sess.run(init)
        data_np = sess.run(op, feed_dict={ data_t: data })
    return data_np

def process_and_write_tfrecord(current_process, imnames, charBBs, txts, writer, coder):
    for i, imname in enumerate(imnames):
        filename = imname[0]
        print('Process {} === {}/{}: Processing {} \n'.format(current_process, i, len(imnames), filename), end='\r')
        img_path = os.path.join(SYNTH_IMG_PATH, 'img', filename)

        # Read to image buffer
        image_buffer, image, height, width, channels = _process_image(img_path, coder)
        image = image[:,:,::-1]
        
        # character bounding box
        char_bbs = charBBs[i] #shape: (2, 4, ?)

        # character text
        txt = txts[i]
        word_texts = txt.copy()
        word_texts = [word_text.strip() for word_text in word_texts]    # ['clusiaceae', '3HC', 'achimenes', 'カルラ']

        # Get weight_character and weight_affinity
        image, character = resize(image, char_bbs.copy())
        chw_image = (image.copy()).transpose(2, 0, 1)
        weight_character = generate_target(chw_image.shape, character.copy())
        weight_affinity, affinity_bbox = generate_affinity(chw_image.shape, character.copy(), word_texts.copy())
        weight_character_buffer = np.clip((weight_character * 255).astype(np.uint8), 0, 255)
        weight_affinity_buffer = np.clip((weight_affinity * 255).astype(np.uint8), 0, 255)

        # print(weight_character_buffer, np.min(weight_character_buffer), np.max(weight_character_buffer))
        # print(weight_affinity_buffer, np.min(weight_affinity_buffer), np.max(weight_affinity_buffer))
        weight_character_buffer = encode_jpeg(weight_character_buffer)
        weight_affinity_buffer = encode_jpeg(weight_affinity_buffer)

        if AUGUMENT:
            imgs = rand_augment(list([image.copy(), weight_character.copy(), weight_affinity.copy()]))
        else:
            imgs = [image.copy(), weight_character.copy(), weight_affinity.copy()]
        imgs[0] = normalize_mean_variance(imgs[0])
        # print(np.unique(imgs[0]))
        # print(np.unique(imgs[1]))
        # print(np.max(imgs[2]))
        image_data = [img.flatten().tolist() for img in imgs]
        example = convert_to_example(img_path, filename, image_buffer, weight_character_buffer, weight_affinity_buffer, char_bbs, word_texts,
                                height, width, channels)
        # print(image_data[0].shape)
        # print(image_data[1].shape)
        # print(image_data[2].shape)

        # break
        writer.write(example.SerializeToString())
    writer.close()

if __name__ == "__main__":
    # Create a generic TensorFlow-based utility for converting all image codings.
    coder = ImageCoder()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    mat = loadmat(os.path.join(SYNTH_IMG_PATH, 'bg.mat'))
    imnames = mat['imnames'][0] # ['file1.png', 'file2.png', ..., 'finen.png']
    charBBs = mat['charBB'][0]  # number of images, 2, 4, num_character
    txts = mat['txt'][0]

    # multiprocessing
    indexes = np.arange(len(imnames))
    chunks = np.array_split(indexes, NUMBER_OF_THREAD)

    # writer = tf.python_io.TFRecordWriter('/hdd/namdng/tf_record_dataset/synth_text/{}_{}.tfrecord'.format(DATA_MODE, 0))
    # process_and_write_tfrecord(0, imnames[chunks[0]], charBBs[chunks[0]], txts[chunks[0]], writer, coder)
    # processes = []
    # for index in range(NUMBER_OF_THREAD):
    #     writer = tf.python_io.TFRecordWriter('/hdd/namdng/tf_record_dataset/synth_text/{}_{}.tfrecord'.format(DATA_MODE, index))
    #     p = multiprocessing.Process(target=process_and_write_tfrecord, args=(index, imnames[chunks[index]], charBBs[chunks[index]], txts[chunks[index]], writer, coder))
    #     processes.append(p)

    # for p in processes:
    #     p.start()

    # for p in processes:
    #     p.join()
    
    threads = []
    for index in range(NUMBER_OF_THREAD):
        writer = tf.python_io.TFRecordWriter('/hdd/namdng/tf_record_dataset/synth_text/{}_{}.tfrecord'.format(DATA_MODE, index))
        t = threading.Thread(target=process_and_write_tfrecord, args=(index, imnames[chunks[index]], charBBs[chunks[index]], txts[chunks[index]], writer, coder))
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %
            (datetime.now(), len(imnames)))
    sys.stdout.flush()

    