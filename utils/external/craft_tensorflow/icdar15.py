import os
import cv2
import numpy as np
import random
import config as CONFIG
from data_manipulation import resize
random.seed(10)


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
            corner_coordinates = np.array(elements[:8]).reshape(1, 4, 2)
            print('Special Text', label_path, elements, word)
        else:
            corner_coordinates = np.array(elements[:-1]).reshape(1, 4, 2)
            word = elements[-1].rstrip()

        word_boxes.append(corner_coordinates)
        word_text.append(word)
            
        if word not in CONFIG.DONTCARE_LABEL:
            valid_data = True

    if valid_data:
        return word_boxes, word_text
    else:
        return [], []


class ICDAR15(object):
    def __init__(self, img_dir_path, label_dir_path, batch_size, side=None, shuffle=True):
        self.img_dir = img_dir_path
        self.lbl_dir = label_dir_path
        self.batch_size = batch_size
        self.current_batch_idx = 0
        self.shuffle = shuffle
        self.data = []
        self.side = side

        img_list = os.listdir(self.img_dir)
        img_list.sort()
        lbl_list = os.listdir(self.lbl_dir)
        lbl_list.sort()

        assert len(img_list) == len(lbl_list), 'ASSERT: Length of Img dir and Lbl dir are different'

        for idx, img_filename in enumerate(img_list):
            lbl_filename = lbl_list[idx]

            if img_filename.split('.')[0] not in lbl_filename:
                continue

            img_path = os.path.join(self.img_dir, img_filename)
            
            label_path = os.path.join(self.lbl_dir, lbl_filename)

            img = cv2.imread(img_path)[:,:,::-1] # Read BGR then converting to RGB image
            word_boxes, word_text = load_icdar15_label(label_path)

            if len(word_text) > 0:
                box_coordinate = np.concatenate(word_boxes, axis=0)
                if self.side is None:
                    self.data.append([img, box_coordinate, word_text, img_filename])
                else:
                    # resize and padding image, then re-calculate new box coordinate
                    img, new_box_coordinate = resize(img, box_coordinate.transpose(2,1,0), side=self.side)
                    self.data.append([img, new_box_coordinate.transpose(2,1,0), word_text, img_filename])


        # print('Length=', len(self.data))

        self.num_batches = len(self.data) // self.batch_size
        self.sample_idx = list(range(len(self.data)))

    def get_batch(self):
        if self.current_batch_idx % self.num_batches == 0:
            self.current_batch_idx = 0
            if self.shuffle is True:
                random.shuffle(self.sample_idx)

        start_idx = self.current_batch_idx * self.batch_size
        end_idx = start_idx + self.batch_size
        batch_sample_idx = self.sample_idx[start_idx:end_idx]
        self.current_batch_idx += 1

        batch_data = [self.data[i] for i in batch_sample_idx]

        return batch_data