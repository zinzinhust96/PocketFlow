import cv2
import numpy as np
import utils.external.craft_tensorflow.config as CONFIG
import math
from shapely.geometry import Polygon
import matplotlib.pyplot as plt # for debug
import traceback # for debug


def transform_word_img_revert(img, corners, target_shape):
    if len(img.shape) == 2:
        h, w = img.shape
    else:
        h, w, _ = img.shape
    target_h, target_w = target_shape
    pts2 = np.float32(corners)
    pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(
        img, M, (target_w, target_h), flags=cv2.INTER_NEAREST)
    return warped  # cv2.warpPerspective(img,M,output_size)


def transform_word_img(img, corners):
    pts1 = np.float32(corners)
    x1, y1 = corners[0]
    x2, y2 = corners[1]
    x3, y3 = corners[2]
    target_w = int(np.sqrt((x2-x1)**2 + (y2-y1)**2))
    target_h = int(np.sqrt((x3-x2)**2 + (y3-y2)**2))
    pts2 = np.float32(
        [[0, 0], [target_w, 0], [target_w, target_h], [0, target_h]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    # print(np.asarray(img))
    text_box = cv2.warpPerspective(np.asarray(img), M, (target_w, target_h))
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

    grid = np.zeros((total_grid, 768, 768, 3), np.uint8)
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


def decay_learning_rate(ite):
    lr = CONFIG.learning_rate * 1.0 / (1.0 + CONFIG.decay_factor_lr*ite)
    return lr


def visual_heatmap(image, heatmap, opa=0.7):
    im_h, im_w = image.shape[:2]
    # cvt heatmap
    image = image.copy()
    if np.max(heatmap) != 0:
        ratio = 255. / np.max(heatmap)
    else:
        ratio = 0

    score_text = np.asarray(heatmap*ratio).astype(np.uint8)
    heatmap = cv2.applyColorMap(score_text, cv2.COLORMAP_JET)

    image = cv2.addWeighted(image, opa, heatmap, 1-opa, 0)

    return image


def make_image_from_batch(X):
    if len(X.shape) == 3:
        X = np.expand_dims(X, -1)

    batch_size, h, w, c = X.shape
    no_col = int(np.ceil(np.sqrt(batch_size)))
    no_row = int(np.ceil(batch_size/no_col))
    output = np.zeros((int(no_row*h), int(no_col*w), c))
    for row in range(no_row):
        for col in range(no_col):
            if (row*no_col + col) == batch_size:
                break
            output[row*h:(row+1)*h, col*w:(col+1)*w] = X[row*no_col + col]

        if (row*no_col + col) == batch_size:
                break
    return np.squeeze(output)



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
    ret, text_score = cv2.threshold(character_heatmap, character_threshold, 1, 0)
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
            seg_map[np.logical_and(link_score == 1, text_score == 0)] = 0  # remove link area

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

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
            seg_map[sy:ey, sx:ex] = cv2.dilate(seg_map[sy:ey, sx:ex], kernel)

            # make box
            np_contours = np.roll(np.array(np.where(seg_map != 0)), 1, axis=0).transpose().reshape(-1, 2)
            rectangle = cv2.minAreaRect(np_contours)
            box = cv2.boxPoints(rectangle)

            if Polygon(box).area == 0:
                continue

            # align diamond-shape
            w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
            box_ratio = max(w, h) / (min(w, h) + 1e-5)
            if abs(1 - box_ratio) <= 0.1:
                l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
                t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
                box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

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

	a = Polygon(poly1.reshape([poly1.shape[0], 2])).buffer(0)
	b = Polygon(poly2.reshape([poly2.shape[0], 2])).buffer(0)

	union_area = a.union(b).area

	if union_area == 0:
		return 0

	return a.intersection(b).area/union_area


def calculate_fscore(pred, target, threshold=0.5):
    """

	:param pred: numpy array with shape [num_words, 4, 2]
	:param target: numpy array with shape [num_words, 4, 2]
	:param threshold: overlap iou threshold over which we say the pair is positive
	:return:
	"""
    if pred.shape[0] == target.shape[0] and target.shape[0] == 0:
        return {
            'f_score': 1.0,
            'precision': 1.0,
            'recall': 1.0,
            'false_positive': 0.0,
            'true_positive': 0.0,
            'num_positive': 0.0
        }
    
    if target.shape[0] == 0:
        return {
            'f_score': 0.0,
            'precision': 0.0,
            'recall': 1.0,
            'false_positive': pred.shape[0],
            'true_positive': 0,
            'num_positive': 0
        }

    if pred.shape[0] == 0:
        return {
            'f_score': 0.0,
            'precision': 1.0,
            'recall': 0.0,
            'false_positive': 0,
            'true_positive': 0,
            'num_positive': target.shape[0]
        }

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
    if precision + recall == 0 :
        return {
            'f_score': 0,
            'precision': precision,
            'recall': recall,
            'false_positive': false_positive,
            'true_positive': true_positive,
            'num_positive': num_positive
        }
    else:
        return {
            'f_score': 2*precision*recall/(precision + recall),
            'precision': precision,
            'recall': recall,
            'false_positive': false_positive,
            'true_positive': true_positive,
            'num_positive': num_positive
        }