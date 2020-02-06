import tensorflow as tf
import sys

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

def preprocess_image(image_buffer, output_side, num_channels, is_training=False):
  if is_training:
    # For training, we want to decode and resize the image.
    image = tf.image.decode_jpeg(image_buffer, channels=num_channels)
    image = _resize_image_with_padding(image, output_side)

  image.set_shape([output_side, output_side, num_channels])

  return image

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
  
  
