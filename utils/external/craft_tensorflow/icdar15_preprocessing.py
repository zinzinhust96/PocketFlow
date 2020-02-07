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
  batch_imgs = []
  lookup_heatmaps = []
  for idx, (image, word_coord, word_text) in enumerate(zip(images, word_coords, word_texts)):
    batch_img, lookup_heatmap = gen_batch_image_contain_only_words(image, zip(word_coord, word_text))
    batch_img = norm_mean_variance(batch_img)
    batch_imgs.append(batch_img)
    lookup_heatmaps.append(lookup_heatmap)
  return np.asarray(batch_imgs), np.asarray(lookup_heatmaps)
  
def create_label(images, weakly_batch, heatmaps, lookup_heatmaps):
  batch_imgs = []
  batch_wc = []
  batch_wa = []
  batch_confi = []
  print(images)
  for idx, (image, batch_img, heatmap, lookup_heatmap) in enumerate(zip(images, weakly_batch, heatmaps, lookup_heatmaps)):
    heat_c = heatmap[:,:,:, 0]
    lookup_word_box = calc_character_boxes(heat_c, lookup_heatmap)
    _, char_heat, affinity_heat, weight_map = create_weakly_heatmaps(image, lookup_word_box, denormalize_mean_variance(batch_img))
    
    normed_img = norm_mean_variance(image)
    batch_imgs.append(normed_img)
    batch_wc.append(char_heat)
    batch_wa.append(affinity_heat)
    batch_confi.append(weight_map)

    return np.asarray(batch_imgs), np.asarray(batch_wc), np.asarray(batch_wa), np.asarray(batch_confi)

def transform_word_img(img, corners):
    pts1 = tf.cast(corners, tf.float32)
    x1, y1 = corners[0]
    x2, y2 = corners[1]
    x3, y3 = corners[2]
    target_w = tf.cast(tf.math.sqrt((x2-x1)**2 + (y2-y1)**2), tf.int32)
    target_h = tf.cast(tf.math.sqrt((x3-x2)**2 + (y3-y2)**2), tf.int32)
    pts2 = tf.constant([[0, 0], [target_w, 0], [target_w, target_h], [0, target_h]], tf.float32)
    M = tf.py_function(cv2.getPerspectiveTransform, [pts1, pts2], (tf.float32))
    # M = cv2.getPerspectiveTransform(pts1, pts2)
    # print(np.asarray(img))
    text_box = tf.py_function(cv2.warpPerspective, [img, M, (target_w, target_h)], (tf.float32))
    # text_box = cv2.warpPerspective(np.asarray(img), M, (target_w, target_h))
    # cv2.warpPerspective(img,M,output_size)
    return text_box, target_w, target_h

def gen_batch_image_contain_only_words(img, boxes):
    all_word_boxes = []
    max_h_box = 0
    max_w_box = 0
    for corners, word in boxes:
        word_img, target_w, target_h = transform_word_img(img, corners)
        if target_h > max_h_box:
            max_h_box = target_h
        if target_w > max_w_box:
            max_w_box = target_w

        all_word_boxes.append([word_img, word, corners])
    # generate grid
    max_w_box += 10
    max_h_box += 10
    num_cols = 768 // max_w_box
    num_rows = 768 // max_h_box
    total_text_box_per_grid = num_cols*num_rows
    total_grid = len(all_word_boxes) // total_text_box_per_grid
    if len(all_word_boxes) % total_text_box_per_grid != 0:
        total_grid += 1

    grid = tf.zeros((total_grid, 768, 768, 3), tf.uint8)
    lookup_heatmap = []
    for id_grid in range(total_grid):
        for id_row in range(num_rows):
            for id_col in range(num_cols):
                if len(all_word_boxes) == 0:
                    break
                text_box, text, pos_in_origin_img = all_word_boxes[0]
                del all_word_boxes[0]
                h, w, _ = text_box.shape
                offset_x = id_col*max_w_box
                offset_y = id_row*max_h_box
                grid[id_grid, offset_y:offset_y+h,
                    offset_x:offset_x+w] = text_box
                lookup_heatmap.append(
                    [id_grid, offset_x, offset_y, offset_x+w,
                        offset_y+h, text, pos_in_origin_img]
                )  # pos_in_origin_img: 4 points, shape 4x2
    return grid, lookup_heatmap


# def create_weakly_batch(images, word_coords, word_texts):
#   for idx, (image, word_coord, word_text) in enumerate(zip(images, word_coords, word_texts)):
#     batch_img, lookup_heatmap = gen_batch_image_contain_only_words(image, zip(word_coord, word_text))
#     batch_img = norm_mean_variance(batch_img)
#     print(type(batch_img))
#     print(batch_img.shape)
#     heat_map = CRAFT(batch_img, False)
#     heat_c = heat_map[:,:,:, 0]
#     lookup_word_box = calc_character_boxes(heat_c, lookup_heatmap) # each element of lookup_word_box contains [id_grid, x1_word, y1_word, x2_word, y2_word, text, pos_in_origin_img, boxes_char, confident_score]
#                                                                        # in which boxes_char is list of character boxes in the word box
#     _, char_heat, affinity_heat, weight_map = create_weakly_heatmaps(image, lookup_word_box, denormalize_mean_variance(batch_img))
    
#     normed_img = norm_mean_variance(image)
#     batch_imgs.append(normed_img)
#     batch_wc.append(char_heat)
#     batch_wa.append(affinity_heat)
#     batch_confi.append(weight_map)

#   return np.asarray(batch_imgs), batch_wc, batch_wa, np.asarray(batch_confi)