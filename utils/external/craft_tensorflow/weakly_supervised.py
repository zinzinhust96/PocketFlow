import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from math import fabs
import random
from utils.data_manipulation import generate_target, generate_affinity
from utils.utils import transform_word_img_revert
import config as CONFIG


def watershed_segmen(heat_box, min_distance):
    ratio = 255. / np.max(heat_box)
    heat_box = np.clip(np.asarray(heat_box*ratio),0,255).astype(np.uint8)
    heat_box = cv2.GaussianBlur(heat_box,(3,3),0)

    binary = np.where(heat_box>0.35*255, 255, 0)

    # min_distance = max(x2-x1, y2-y1)*0.7/len(text)
    local_maxi = peak_local_max(heat_box, indices=False,footprint=np.ones((3, 3)), min_distance=min_distance,
                                labels=binary)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-heat_box, markers, mask=binary)
    return labels


def calc_confident_score(gt_len, pred_len):
    return (gt_len - min(gt_len, fabs(gt_len - pred_len))) / gt_len


def extract_character_box(heat_word, text):
    ori_h, ori_w = heat_word.shape[:2]
    min_distance = max(ori_h, ori_w)*0.7/len(text)
    labels = watershed_segmen(heat_word, min_distance)
    # devide box
    boxes = []
    
    centroids = []
    if np.max(labels) == 0:
        return boxes, 'horizontal' # default is horizontal

    for i in range(1,np.max(labels)+1):
        xs,ys = np.where(labels == i)
        y1, y2 = np.min(xs), np.max(xs)
        x1, x2 = np.min(ys), np.max(ys)
        cen = [(x2+x1) // 2, (y2+y1)//2]
        # w = x2-x1
        # h = y2-y1
        centroids.append(cen)
        boxes.append(np.array([x1,y1,x2,y2]))

    centroids = np.stack(centroids)
    (x,y,w,h) = cv2.boundingRect(centroids)

    if w >= h: # check overlap in x axis
        boxes = sorted(boxes, key=lambda x:x[0]) # sort by x1

        type_word = 'horizontal'

        id_merged_box = []
        boxes_refine = []
        for id_box, (x1,y1,x2,y2) in enumerate(boxes):
            x1_rf, y1_rf, x2_rf, y2_rf = x1,y1,x2,y2

            if id_box == len(boxes) or id_box in id_merged_box:
                continue
            for id_box_rest, (x1_,y1_,x2_,y2_) in enumerate(boxes[id_box+1:]):
                if x1_ < x2: # overlap
                    if x2-x1_ >= 0.5 * min(x2-x1, x2_-x1_):
                        # merge

                        id_merged_box.append(id_box + 1 + id_box_rest)
                        x2_rf = max(x2_,x2_rf)
                        y1_rf = min(y1_rf, y1_)
                        y2_rf = max(y2_rf, y2_)
                else:
                    break
            # print(x1,y1,x2,y2)
            # print(x1_rf, y1_rf, x2_rf, y2_rf)
            boxes_refine.append([x1_rf, y1_rf, x2_rf, y2_rf])
    else: # check overlap in y axis
        boxes = sorted(boxes, key=lambda x:x[1]) # sort by y1
        type_word = 'vertical'
        
        id_merged_box = []
        boxes_refine = []
        for id_box, (x1,y1,x2,y2) in enumerate(boxes):
            x1_rf, y1_rf, x2_rf, y2_rf = x1,y1,x2,y2
            if id_box == len(boxes) or id_box in id_merged_box:
                continue
            for id_box_rest, (x1_,y1_,x2_,y2_) in enumerate(boxes[id_box+1:]):
                if y1_ < y2: # overlap
                    if y2-y1_ >= 0.5 * min(y2-y1, y2_-y1_):
                        # merge
                        id_merged_box.append(id_box + 1 + id_box_rest)
                        y2_rf = max(y2_,y2_rf)
                        x1_rf = min(x1_rf, x1_)
                        x2_rf = max(x2_rf, x2_)
                else:
                    break
            boxes_refine.append([x1_rf, y1_rf, x2_rf, y2_rf])

    # merge boxes
    boxes_char = []
    if type_word == 'horizontal':
        avg_width_char = ori_w / len(text)
        for x1,y1,x2,y2 in (boxes_refine):
            if x2-x1 > 0.4*avg_width_char:
                boxes_char.append([x1,y1,x2,y2])
    else: # vertical
        avg_width_char = ori_h / len(text)
        for x1,y1,x2,y2 in (boxes_refine):
            if y2-y1 > 0.4*avg_width_char:
                boxes_char.append([x1,y1,x2,y2])

    # remove some small boxes
    return boxes_char, type_word


def recheck_detected_char_boxes(boxes_char, type_word, text, box_h, box_w):
    conf_scr = calc_confident_score(len(text), len(boxes_char))

    # check total area 
    boxes_char = np.array(boxes_char)
    total_area_char = 0
    for x1,y1,x2,y2 in boxes_char:
        total_area_char += (x2-x1)*(y2-y1)

    is_satisfied_area = True
    if total_area_char <= 0.4 * (box_h * box_w):
        is_satisfied_area = False

    is_satisfied_length = True
    if len(boxes_char) == 0:
        is_satisfied_length = False
    else:
        x_min = np.min(boxes_char[:,0])
        x_max = np.max(boxes_char[:,2])
        y_min = np.min(boxes_char[:,1])
        y_max = np.max(boxes_char[:,3])

        if type_word == 'horizontal':
            if x_max - x_min <= 0.8 * box_w:
                is_satisfied_length = False
        else:
            if y_max - y_min <= 0.8 * box_h:
                is_satisfied_length = False

    if (not is_satisfied_area) or (not is_satisfied_length) or (conf_scr <= 0.5):
        conf_scr = 0.5

        boxes_char = []
        if type_word == 'horizontal':
            w_char = box_w // len(text)
            mod = box_w % len(text)
            current_x = 0
            for i in range(len(text)):
                boxes_char.append([current_x, 0, current_x + w_char, box_h])
                current_x = current_x + w_char
                if mod > 0:
                    mod -= 1
                    current_x += 1
        else:    
            w_char = box_h // len(text)
            mod = box_h % len(text)
            current_y = 0
            for i in range(len(text)):
                boxes_char.append([0, current_y, box_w, current_y+w_char])
                current_y = current_y + w_char
                if mod > 0:
                    mod -= 1
                    current_y += 1

    return boxes_char, conf_scr


def calc_character_boxes(character_heatmap, lookup_heatmap):
    lookup_word_box = []

    for id_grid, x1_word, y1_word, x2_word, y2_word, text, pos_in_origin_img in lookup_heatmap:
        # print('Processing ', text)
        if text == '###':
            lookup_word_box.append([id_grid, x1_word, y1_word, x2_word, y2_word, text, pos_in_origin_img, [], 0])
            continue
        
        # print(character_heatmap.shape)
        heat_box = character_heatmap[id_grid,y1_word:y2_word,x1_word:x2_word].copy()
        # print(extract_character_box(heat_box, text))
        boxes_char, type_word = extract_character_box(heat_box, text)

        boxes_char, confident_score = recheck_detected_char_boxes(boxes_char, type_word, text, y2_word - y1_word, x2_word - x1_word)

        lookup_word_box.append([id_grid, x1_word, y1_word, x2_word, y2_word, text, pos_in_origin_img, boxes_char, confident_score])

    return lookup_word_box


def custom_resize(image, side=768):
    height, width, channel = image.shape
    max_side = max(height, width)
    new_resize = (int(width/max_side*side), int(height/max_side*side))
    image = cv2.resize(image, new_resize)
    new_h, new_w = image.shape[:2]

    y_start = (side - new_h)//2
    x_start = (side - new_w)//2

    big_image = np.ones([side, side, 3], dtype=np.float32)*np.mean(image)
    big_image[
        y_start: y_start + new_h,
        x_start: x_start + new_w] = image
    big_image = big_image.astype(np.uint8)

    return big_image, new_resize, x_start, y_start


def create_weakly_heatmaps(img, lookup_word_box, grid_img):
    # create character heatmap and affinity heatmap
    target_shape = img.shape[:2]

    character_heatmap = np.zeros(shape=target_shape)
    affinity_heatmap = np.zeros(shape=target_shape)
    weight_map = np.zeros(shape=target_shape)

    for id_grid, x1_word, y1_word, x2_word, y2_word, text, pos_in_origin_img, boxes_refine, confident_score in lookup_word_box:
        char_heat = np.zeros(shape=(y2_word - y1_word, x2_word - x1_word))
        affi_heat = np.zeros(shape=(y2_word - y1_word, x2_word - x1_word))
        # print(text)

        # Determine the smallest retangle that covers the original text box
        pos_in_origin_img = pos_in_origin_img.astype(np.int)
        xmin, xmax = np.min(pos_in_origin_img[:, 0]), np.max(pos_in_origin_img[:, 0])
        ymin, ymax = np.min(pos_in_origin_img[:, 1]), np.max(pos_in_origin_img[:, 1])

        if text == '###':
            weight = np.ones_like(affi_heat, dtype=np.float32) * -1000
            revert_weight = transform_word_img_revert(weight, pos_in_origin_img, target_shape)
            weight_map[ymin:ymax, xmin:xmax][revert_weight[ymin:ymax, xmin:xmax] == -1000] = -1000
            continue

        # convert box from [n, 4] to (2, 4, n)
        boxes_char = [np.array([[x1,y1], [x2,y1], [x2,y2], [x1,y2]]) for x1,y1,x2,y2 in boxes_refine]

        boxes_char = np.stack(boxes_char).transpose(2, 1, 0)
        heat_box_shape = (3, y2_word-y1_word, x2_word-x1_word)

        # Generate character heatmap
        char_heat = generate_target(heat_box_shape, boxes_char.copy())

        # Generate affinity heatmap
        text_fake = [''.join(['a'] * boxes_char.shape[-1])]
        affi_heat, affinity_bbox = generate_affinity(heat_box_shape, boxes_char.copy(), text_fake)

        weight = np.ones_like(affi_heat, dtype=np.float32) * confident_score
        # cv2.imwrite('debug/heat_{}_{}_{}.png'.format(text, pos_in_origin_img[0][0], pos_in_origin_img[0][1]), np.hstack((char_heat, affi_heat)) * 255)

        revert_char_heat = transform_word_img_revert(char_heat, pos_in_origin_img, target_shape)
        revert_affi_heat = transform_word_img_revert(affi_heat, pos_in_origin_img, target_shape)
        revert_weight = transform_word_img_revert(weight, pos_in_origin_img, target_shape)

        character_heatmap[ymin:ymax, xmin:xmax] = np.maximum(character_heatmap[ymin:ymax, xmin:xmax], revert_char_heat[ymin:ymax, xmin:xmax])
        affinity_heatmap[ymin:ymax, xmin:xmax] = np.maximum(affinity_heatmap[ymin:ymax, xmin:xmax], revert_affi_heat[ymin:ymax, xmin:xmax])
        weight_map[ymin:ymax, xmin:xmax] = np.maximum(weight_map[ymin:ymax, xmin:xmax], revert_weight[ymin:ymax, xmin:xmax])
    
    weight_map[np.where(weight_map == 0)] = 1
    weight_map[np.where(weight_map < 0)] = 0

    return img, character_heatmap, affinity_heatmap, weight_map

