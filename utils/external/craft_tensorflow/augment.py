import os
import numpy as np
import cv2
import random

def crop_img(src,top_left_x,top_left_y,crop_w,crop_h):
    '''Crop image
    Args:
        src: Source image
        top_left_x,top_left_y: Upper left coordinate of cropped image (on Source image)
        crop_w,crop_h：Width and Hight of cropped image
    return：
        cropped_img: cropped image
        None: Input is invalied
    '''
    rows, cols = src.shape[0: 2]
    row_min,col_min = int(top_left_y), int(top_left_x)
    row_max,col_max = int(row_min + crop_h), int(col_min + crop_w)
    if row_max > rows or col_max > cols:
        print("crop size err: src->%dx%d,crop->top_left(%d,%d) %dx%d"%(cols, rows, col_min, row_min,int(crop_w),int(crop_h)))
        return None
    cropped_img = src[row_min:row_max, col_min:col_max].copy()
    return cropped_img

def crop_imgs(imgs, crop_type='RANDOM_CROP', dsize=(0, 0), random_wh=False):
    '''
    Args：
        imgs: [image, weight_character, weight_affinity, normal_image]
        crop_type: Cropping type ['RANDOM_CROP','CENTER_CROP','FIVE_CROP']
        dsize: Specify crop width and height（w,h），Mutual exclusion with random_wh == True
        random_wh：Randomly select crop width and height
    '''
    imgh, imgw = imgs[0].shape[0: 2]
    # fw, fh: Crop ratio when random_wh == False, otherwise the lower limit of width and height ratio of random crop

    fw = random.uniform(0.2, 0.98)
    fh = random.uniform(0.2, 0.98)
    crop_imgw, crop_imgh = dsize
    if dsize == (0, 0) and not random_wh:
        crop_imgw = int(imgw * fw)
        crop_imgh = int(imgh * fh)
    elif random_wh:
        crop_imgw = int(imgw * (fw + random.random() * (1 - fw)))
        crop_imgh = int(imgh * (fh + random.random() * (1 - fh)))

    if crop_type == 'RANDOM_CROP':
        crop_top_left_x, crop_top_left_y = random.randint(0, imgw - crop_imgw - 1), random.randint(0, imgh - crop_imgh - 1)
    elif crop_type == 'CENTER_CROP':
        crop_top_left_x, crop_top_left_y = int(imgw / 2 - crop_imgw / 2), int(imgh / 2 - crop_imgh / 2)
    elif crop_type == 'FIVE_CROP':
        crop_top_left_x, crop_top_left_y = 0, 0
    else:
        print('crop type wrong! expect [RANDOM_CROP,CENTER_CROP,FIVE_CROP]')

    ret_imgs = []
    original_img_size = (imgw, imgh)

    for idx, img in enumerate(imgs):
        # print('Img shape before cropping', img.shape)
        cropped_img = crop_img(img, crop_top_left_x, crop_top_left_y, crop_imgw, crop_imgh)
        if cropped_img is None:
            return imgs

        if idx in [1, 2]: # Only check for Character Heatmap and Affinity Heatmap
            # Discard fewer positive samples
            tmp_original = img.copy()
            tmp_original[tmp_original > 0] = 1
            num_active_pixels_original_img = np.sum(tmp_original)

            tmp_cropped = cropped_img.copy()
            tmp_cropped[tmp_cropped > 0] = 1
            num_active_pixels_cropped_img = np.sum(tmp_cropped)

            if num_active_pixels_cropped_img / num_active_pixels_original_img < 0.5: # we will ignore that cropped image if number of active pixels after cropping < 0.5 at original image
                return imgs

        cropped_img = cv2.resize(cropped_img, original_img_size)
        ret_imgs.append(cropped_img)

    return ret_imgs

def rot_img_and_padding(img, rot_angle, scale=1.0):
    '''
    Rotate the center of the picture
    Args:
        img: Source image
        rot_angle: Rotation angle, counterclockwise
        scale: Scale
    return:
        imgRotation: picture after rotation
    '''
    img_rows, img_cols = img.shape[:2]
    cterxy = [img_cols//2, img_rows//2]

    matRotation = cv2.getRotationMatrix2D((cterxy[0], cterxy[1]), rot_angle, scale)
    imgRotation = cv2.warpAffine(img, matRotation, (img_cols, img_rows))
    return imgRotation

def rand_rot(imgs):
    '''
    Randomly rotate images
    :param imgs: [image, weight_character, weight_affinity, normal_image]
    :return:
    '''
    angle = random.randint(0, 180)
    scale = random.uniform(0.9, 1.5)
    ret_imgs = []

    for img in imgs:
        ret_imgs.append(rot_img_and_padding(img, angle, scale))

    return ret_imgs

def rand_flip(imgs):
    '''Flip image'''
    flag = random.random()
    ret_imgs = []

    if flag < 0.3333:
        for img in imgs:
            ret_imgs.append(cv2.flip(img, 1))

    elif (flag >= 0.3333) and (flag < 0.6666):
        for img in imgs:
            ret_imgs.append(cv2.flip(img, -1))

    else:
        for img in imgs:
            ret_imgs.append(cv2.flip(img, 0))
            
    return ret_imgs
    

def random_color_distort(img, brightness_delta=32, hue_vari=18, sat_vari=0.5, val_vari=0.5):
    '''
    randomly distort image color. Adjust brightness, hue, saturation, value.
    param:
        img: a BGR uint8 format OpenCV image. HWC format.
    '''

    def random_hue(img_hsv, hue_vari, p=0.5):
        if np.random.uniform(0, 1) > p:
            hue_delta = np.random.randint(-hue_vari, hue_vari)
            img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
        return img_hsv

    def random_saturation(img_hsv, sat_vari, p=0.5):
        if np.random.uniform(0, 1) > p:
            sat_mult = 1 + np.random.uniform(-sat_vari, sat_vari)
            img_hsv[:, :, 1] *= sat_mult
        return img_hsv

    def random_value(img_hsv, val_vari, p=0.5):
        if np.random.uniform(0, 1) > p:
            val_mult = 1 + np.random.uniform(-val_vari, val_vari)
            img_hsv[:, :, 2] *= val_mult
        return img_hsv

    def random_brightness(img, brightness_delta, p=0.5):
        if np.random.uniform(0, 1) > p:
            img = img.astype(np.float32)
            brightness_delta = int(np.random.uniform(-brightness_delta, brightness_delta))
            img = img + brightness_delta
        return np.clip(img, 0, 255)

    # brightness
    img = random_brightness(img, brightness_delta)
    img = img.astype(np.uint8)

    # color jitter
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    if np.random.randint(0, 2):
        img_hsv = random_value(img_hsv, val_vari)
        img_hsv = random_saturation(img_hsv, sat_vari)
        img_hsv = random_hue(img_hsv, hue_vari)
    else:
        img_hsv = random_saturation(img_hsv, sat_vari)
        img_hsv = random_hue(img_hsv, hue_vari)
        img_hsv = random_value(img_hsv, val_vari)

    img_hsv = np.clip(img_hsv, 0, 255) # Limitting
    img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR) # Convert color space

    return img

def tranc(imgs):
    ret_imgs = []
    for img in imgs:
        ret_imgs.append(cv2.transpose(img))

    return ret_imgs

def rand_augment(imgs):
    '''
    Randomly select a data 
    Args:
        imgs: [image, weight_character, weight_affinity, normal_image]
    Return:
        ret_imgs: augmented images
    '''

    ret_imgs = None
    if random.random() < 0.5:
        # Random crop
        ret_imgs = crop_imgs(imgs)
        if random.random() < 0.5:
            ret_imgs = tranc(ret_imgs)

    elif random.random() < 0.5:
        # Random flip
        ret_imgs = rand_flip(imgs)

        if random.random() < 0.5:
            ret_imgs = tranc(ret_imgs)
 
    elif random.random() < 0.5:
        # Random chroma transform
        ret_imgs = []
        ret_imgs.append(random_color_distort(imgs[0]))
        ret_imgs += imgs[1:]
        # ret_imgs.append(imgs[1])
        # ret_imgs.append(imgs[2])
        # ret_imgs.append(imgs[3]) # we don't need to process for normal image because it is only for visulization

        if random.random() < 0.5:
            ret_imgs = tranc(ret_imgs)
    else:
        ret_imgs = imgs

    return ret_imgs
