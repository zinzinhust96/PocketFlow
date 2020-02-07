import sys
import tensorflow as tf
from utils.external.craft_tensorflow.data_manipulation import normalize_mean_variance as norm_mean_variance
from utils.external.craft_tensorflow.utils import gen_batch_image_contain_only_words
from utils.external.craft_tensorflow.weakly_supervised import calc_character_boxes, create_weakly_heatmaps
from utils.external.craft_tensorflow.model import craft as CRAFT

def _resize_image_with_padding(image, side):
  """Simple wrapper around tf.image.resize_image_with_pad.

  Args:
    image: A 3-D image `Tensor`.
    side: The target side for the resized image.

  Returns:
    resized_image: A 3-D tensor containing the resized image. The first two
      dimensions have the shape [side, side].
  """
  return tf.image.resize_image_with_pad(image, side, side)

def normalize_mean_variance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
  # should be RGB order
  img = tf.cast(in_img, tf.float32)
  img -= tf.constant([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=tf.float32)
  img /= tf.constant([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=tf.float32)
  return img

def preprocess_image(image_buffer, output_side, num_channels, is_training=False):
  # if is_training:
    # For training, we want to decode and resize the image.
  image = tf.image.decode_jpeg(image_buffer, channels=num_channels)
  image = _resize_image_with_padding(image, output_side)
  image.set_shape([output_side, output_side, num_channels])

  return normalize_mean_variance(image)

def preprocess_label(input_shape, character, side):
  width, height, channel = input_shape
  max_side = tf.math.maximum(height, width)
  new_resize = (tf.cast(width/max_side*side, tf.int32), tf.cast(height/max_side*side, tf.int32))
  if character is not None:
    x_coords = tf.expand_dims(character[0, :, :], 0)
    y_coords = tf.expand_dims(character[1, :, :], 0)
    x_coords = tf.cast(x_coords, tf.int32)//width*new_resize[0] + (side-width)//2
    y_coords = tf.cast(y_coords, tf.int32)//height*new_resize[1] + (side-height)//2
    character = tf.transpose(tf.concat([x_coords, y_coords], axis=0))
  return character
  
  
def create_weakly_batch(images, word_coords, word_texts):
  for idx, (image, word_coord, word_text) in zip(images, word_coords, word_texts):
    batch_img, lookup_heatmap = gen_batch_image_contain_only_words(image, word_coord, word_text)
    batch_img = norm_mean_variance(batch_img)
    heat_map = CRAFT(batch_img, False)
    heat_c = heat_map[:,:,:, 0]
    lookup_word_box = calc_character_boxes(heat_c, lookup_heatmap) # each element of lookup_word_box contains [id_grid, x1_word, y1_word, x2_word, y2_word, text, pos_in_origin_img, boxes_char, confident_score]
                                                                       # in which boxes_char is list of character boxes in the word box
    _, char_heat, affinity_heat, weight_map = create_weakly_heatmaps(image, lookup_word_box, denormalize_mean_variance(batch_img))
    
    normed_img = norm_mean_variance(image)
    batch_imgs.append(normed_img)
    batch_wc.append(char_heat)
    batch_wa.append(affinity_heat)
    batch_confi.append(weight_map)

  return np.asarray(batch_imgs), batch_wc, batch_wa, np.asarray(batch_confi)