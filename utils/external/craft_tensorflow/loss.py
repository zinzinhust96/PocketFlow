import tensorflow as tf
import numpy as np
import cv2
import math
from shapely.geometry import Polygon
import sys
import utils.external.craft_tensorflow.config as CONFIG

FLAGS = tf.app.flags.FLAGS


def MSE_OHEM_Loss(output_imgs, target_imgs, confident_maps):
    loss_every_sample = []

    if tf.shape(output_imgs)[0] == tf.constant(FLAGS.batch_size, dtype=tf.int32):
        batch_size = FLAGS.batch_size
    else:
        batch_size = FLAGS.batch_size_eval

    for i in range(batch_size):
        output_img = tf.reshape(output_imgs[i], [-1])
        target_img = tf.reshape(target_imgs[i], [-1])
        conf_map = tf.reshape(tf.stack([confident_maps[i], confident_maps[i]], -1), [-1])
        positive_mask = tf.cast(tf.greater(target_img, CONFIG.threshold_positive), dtype=tf.float32)
        # fix
        sample_loss = tf.square(tf.subtract(output_img, target_img)) * conf_map

        num_all = output_img.get_shape().as_list()[0]
        num_positive = tf.cast(tf.reduce_sum(positive_mask), dtype=tf.int32)

        positive_loss = tf.multiply(sample_loss, positive_mask)
        positive_loss_m = tf.reduce_sum(
            positive_loss)/tf.cast(num_positive, dtype=tf.float32)
        negative_mask = tf.cast(tf.less_equal(
            target_img, CONFIG.threshold_negative), dtype=tf.float32)
        nagative_loss = tf.multiply(sample_loss, negative_mask)  # fix
        # nagative_loss_m = tf.reduce_sum(nagative_loss)/(num_all - num_positive)

        k = num_positive * 3
        #nagative_loss_topk, _ = tf.nn.top_k(nagative_loss, k)
        # tensorflow 1.13存在bug，不能使用以下语句 Orz。。。
        k = tf.cond((k + num_positive) > num_all,
                    lambda: tf.cast((num_all - num_positive), dtype=tf.int32), lambda: k)
        k = tf.cond(k > 0, lambda: k, lambda: k+1)
        nagative_loss_topk, _ = tf.nn.top_k(nagative_loss, k)
        res = tf.cond(k < 10, lambda: tf.reduce_mean(sample_loss),
                      lambda: positive_loss_m + tf.reduce_sum(nagative_loss_topk)/tf.cast(k, dtype=tf.float32))
        loss_every_sample.append(res)
    return tf.reduce_mean(tf.convert_to_tensor(loss_every_sample))


def generate_word_bbox(
    character_heatmap,
    affinity_heatmap,
    character_threshold,
    affinity_threshold,
    word_threshold,
    character_threshold_upper,
    affinity_threshold_upper,
    scaling_character,
    scaling_affinity
):
    """
    Given the character heatmap, affinity heatmap, character and affinity threshold this function generates
    character bbox and word-bbox

    :param character_heatmap: Character Heatmap, numpy array, dtype=np.float32, shape = [height, width], value range [0, 1]
    :param affinity_heatmap: Affinity Heatmap, numpy array, dtype=np.float32, shape = [height, width], value range [0, 1]
    :param character_threshold: Threshold above which we say pixel belongs to a character
    :param affinity_threshold: Threshold above which we say a pixel belongs to a affinity
    :param word_threshold: Threshold of any pixel above which we say a group of characters for a word
    :param character_threshold_upper: Threshold above which we differentiate the characters
    :param affinity_threshold_upper: Threshold above which we differentiate the affinity
    :param scaling_character: how much to scale the character bbox
    :param scaling_affinity: how much to scale the affinity bbox
    :return: {
        'word_bbox': word_bbox, type=np.array, dtype=np.int64, shape=[num_words, 4, 1, 2] ,
        'characters': char_bbox, type=list of np.array, dtype=np.int64, shape=[num_words, num_characters, 4, 1, 2] ,
        'affinity': affinity_bbox, type=list of np.array, dtype=np.int64, shape=[num_words, num_affinity, 4, 1, 2] ,
    }
    """

    img_h, img_w = character_heatmap.shape

    """ labeling method """
    ret, text_score = cv2.threshold(
        character_heatmap, character_threshold, 1, 0)
    ret, link_score = cv2.threshold(affinity_heatmap, affinity_threshold, 1, 0)

    text_score_comb = np.clip(text_score + link_score, 0, 1)
    # debug
    # plt.imshow(text_score_comb)
    # plt.show()

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        text_score_comb.astype(np.uint8),
        connectivity=4)

    det = []
    mapper = []
    for k in range(1, n_labels):

        try:
            # size filtering
            size = stats[k, cv2.CC_STAT_AREA]
            if size < 10:
                continue

            where = labels == k

            # thresholding
            if np.max(character_heatmap[where]) < word_threshold:
                continue

            # make segmentation map
            seg_map = np.zeros(character_heatmap.shape, dtype=np.uint8)
            seg_map[where] = 255
            # remove link area
            seg_map[np.logical_and(link_score == 1, text_score == 0)] = 0

            x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
            w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
            niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
            sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
            # boundary check
            if sx < 0:
                sx = 0
            if sy < 0:
                sy = 0
            if ex >= img_w:
                ex = img_w
            if ey >= img_h:
                ey = img_h

            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (1 + niter, 1 + niter))
            seg_map[sy:ey, sx:ex] = cv2.dilate(seg_map[sy:ey, sx:ex], kernel)

            # make box
            np_contours = np.roll(
                np.array(np.where(seg_map != 0)), 1, axis=0).transpose().reshape(-1, 2)
            rectangle = cv2.minAreaRect(np_contours)
            box = cv2.boxPoints(rectangle)

            if Polygon(box).area == 0:
                continue

            # align diamond-shape
            w, h = np.linalg.norm(
                box[0] - box[1]), np.linalg.norm(box[1] - box[2])
            box_ratio = max(w, h) / (min(w, h) + 1e-5)
            if abs(1 - box_ratio) <= 0.1:
                l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
                t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
                box = np.array(
                    [[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

            # make clock-wise order
            start_idx = box.sum(axis=1).argmin()
            box = np.roll(box, 4 - start_idx, 0)
            box = np.array(box)

            det.append(box)
            mapper.append(k)

        except:
            # ToDo - Understand why there is a ValueError: math domain error in line
            #  niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
            print(traceback.format_exc())
            continue

    return np.array(det, dtype=np.int32).reshape([len(det), 4, 2])


def calc_iou(poly1, poly2):
    """
    Function to calculate IOU of two bbox

    :param poly1: numpy array containing co-ordinates with shape [num_points, 1, 2] or [num_points, 2]
    :param poly2: numpy array containing co-ordinates with shape [num_points, 1, 2] or [num_points, 2]
    :return: float representing the IOU
    """
    poly1 = np.asarray(poly1)
    poly2 = np.asarray(poly2)
    a = Polygon(poly1.reshape([-1, 2])).buffer(0)
    b = Polygon(poly2.reshape([-1, 2])).buffer(0)

    union_area = a.union(b).area

    if union_area == 0:
        return 0

    return a.intersection(b).area/union_area


def calculate_fscore(weight_characters, weight_affinitys, target, threshold=0.5):
    """
    :param pred: numpy array with shape [num_words, 4, 2]
    :param target: numpy array with shape [num_words, 4, 2]
    :param threshold: overlap iou threshold over which we say the pair is positive
    :return:
    """
    weight_characters = np.squeeze(weight_characters)
    weight_affinitys = np.squeeze(weight_affinitys)
    pred = generate_word_bbox(weight_characters, weight_affinitys,
                              CONFIG.threshold_character, CONFIG.threshold_affinity,
                              CONFIG.threshold_word, CONFIG.threshold_character_upper,
                              CONFIG.threshold_affinity_upper, CONFIG.scale_character,
                              CONFIG.scale_affinity)

    if pred.shape[0] == target.shape[0] and target.shape[0] == 0:
        return 1.0

    if target.shape[0] == 0:
        print('target.shape[0] == 0:')
        return 0.0

    if pred.shape[0] == 0:
        print('pred.shape[0] == 0')
        return 0.0

    already_done = np.zeros([len(target)], dtype=np.bool)
    false_positive = 0
    for no, i in enumerate(pred):
        found = False
        for j in range(len(target)):
            if already_done[j]:
                continue
            iou = calc_iou(i, target[j])
            if iou >= threshold:
                already_done[j] = True
                found = True
                break
        if not found:
            false_positive += 1
    true_positive = np.sum(already_done.astype(np.float32))
    num_positive = len(target)
    precision = true_positive/(true_positive + false_positive)
    recall = true_positive / num_positive
    if precision + recall == 0:
        print('precision + recall')
        return 0.0
    else:
        print('normal case')
        return 2*precision*recall/(precision + recall)
