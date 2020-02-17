import numpy as np 
import threading
import cv2
import utils.external.craft_tensorflow.config as config
from scipy.io import loadmat
import time
from multiprocessing.pool import ThreadPool
import os
import matplotlib.pyplot as plt
from utils.external.craft_tensorflow.augment import rand_augment

np.random.seed(config.seed)

from utils.external.craft_tensorflow.data_manipulation import resize, normalize_mean_variance, generate_affinity, generate_target, denormalize_mean_variance

class Generator(threading.Thread):
    def __init__(self, file_mat, prefix_path, batch_size=8, augument=True, len_queue=10, num_workers=8, shuffle=True, drop_last=True):
        '''
        file_mat: path to file mat
        prefix_path: path to folder contain images
        '''
        threading.Thread.__init__(self)
        mat = loadmat(file_mat)
        self.total_number = mat['imnames'][0].shape[0]
        self.imnames = mat['imnames'][0] # ['file1.png', 'file2.png', ..., 'finen.png']
        self.charBB = mat['charBB'][0]  # number of images, 2, 4, num_character
        self.txt = mat['txt'][0]
        self.queue = []
        self.len_queue = len_queue
        self.stop_thread = False
        self.num_workers = num_workers
        self.idx_arr = np.arange(self.total_number)
        self.batch_idx = 0
        self.bs = batch_size
        self.num_batches = self.total_number // self.bs
        if (self.num_batches * self.bs) < self.total_number and drop_last:
            self.num_batches += 1
        self.shuffle = shuffle
        self.prefix_path = prefix_path
        self.drop_last = drop_last
        self.augument = augument
        if shuffle:
            np.random.shuffle(self.idx_arr)

    def run(self):
        while 1:
            time.sleep(0.01)
            if self.stop_thread:
                return
            if len(self.queue) >= self.len_queue:
                continue
            self.append_queue()

    def kill(self):
        self.stop_thread = True

    def append_queue(self):
        if self.batch_idx + 1 == self.num_batches:
            self.batch_idx = 0
            if self.shuffle:
                np.random.shuffle(self.idx_arr)

        idx = self.batch_idx*self.bs
        idx = self.idx_arr[idx:idx+self.bs]
        paths = self.imnames[idx] # batch fn
        txts = self.txt[idx] # batch txt
        charBBs = self.charBB[idx] # batch charbb

        with ThreadPool(processes=self.num_workers) as p:
            batch = p.map(self.get_sample, zip(paths, txts, charBBs))

        self.queue.append(batch)
        self.batch_idx += 1

        return
    
    def get_sample(self, arg):
        path, txt, charBB = arg
        txt = [item.strip() for item in txt]
        path = os.path.join(self.prefix_path, path[0]) 
        image = cv2.imread(path)  # Read the image -> BGR image

        if len(image.shape) == 2:
            image = np.repeat(image[:, :, None], repeats=3, axis=2)
        elif image.shape[2] == 1:
            image = np.repeat(image, repeats=3, axis=2)
        else:
            image = image[:, :, 0: 3]

        image, character = resize(image, charBB.copy())  # Resize the image to (768, 768)
        # normal_image = image.astype(np.uint8).copy()
        chw_image = (image.copy()).transpose(2, 0, 1)

        # Generate character heatmap
        # time_1 = time.time()
        weight_character = generate_target(chw_image.shape, character.copy())
        # Generate affinity heatmap
        # time_2 = time.time()
        weight_affinity, affinity_bbox = generate_affinity(chw_image.shape, character.copy(), txt.copy())
        # time_3 = time.time()
        # print('time create w_c: {} | time create w_a: {}'.format(time_2 - time_1, time_3 - time_2))

        # cv2.drawContours(
        #     normal_image,
        #     np.array(affinity_bbox).reshape([len(affinity_bbox), 4, 1, 2]).astype(np.int64), -1, (0, 255, 0), 2)

        # enlarged_affinity_bbox = []

        # for i in affinity_bbox:
        #     center = np.mean(i, axis=0)
        #     i = i - center[None, :]
        #     i = i*60/25
        #     i = i + center[None, :]
        #     enlarged_affinity_bbox.append(i)

        # cv2.drawContours(
        #     normal_image,
        #     np.array(enlarged_affinity_bbox).reshape([len(affinity_bbox), 4, 1, 2]).astype(np.int64),
        #     -1, (0, 0, 255), 2
        # )

        if self.augument:
            # imgs = rand_augment(list([image.copy(), weight_character.copy(), weight_affinity.copy(), normal_image.copy()]))
            imgs = rand_augment(list([image.copy(), weight_character.copy(), weight_affinity.copy()]))
        else:
            imgs = [image.copy(), weight_character.copy(), weight_affinity.copy()]
        
        imgs[0] = normalize_mean_variance(imgs[0][:,:,::-1]) # change from BGR to RGB image

        # return \
		# 	[imgs[0].astype(np.float32), \
		# 	imgs[1].astype(np.float32), \
		# 	imgs[2].astype(np.float32), \
		# 	imgs[3]]
        return \
        [imgs[0].astype(np.float32), \
        imgs[1].astype(np.float32), \
        imgs[2].astype(np.float32)]
    
    def get_batch(self):
        while 1:
            if len(self.queue) == 0:
                time.sleep(0.1)
                # print('empty queue, just wait for a second')
                continue
            else:
                break
        
        first_batch = self.queue[0]
        batch_image = []
        batch_weight_character = []
        batch_weight_affinity = []
        # batch_normal_image = []

        for sample in first_batch:
            batch_image.append(sample[0])
            batch_weight_character.append(sample[1])
            batch_weight_affinity.append(sample[2])
            # batch_normal_image.append(sample[3])

        image = np.array(batch_image)
        weight_character = np.array(batch_weight_character)
        weight_affinity = np.array(batch_weight_affinity)
        # normal_image = np.array(batch_normal_image)
        del self.queue[0]

        return image, weight_character, weight_affinity#, normal_image

if __name__ == "__main__":

    gen = Generator(
        file_mat='/hdd/UBD/background-images/data_generated/29_11_close_line/val/bg.mat', 
        prefix_path='/hdd/UBD/background-images/data_generated/29_11_close_line/val/img',
        len_queue=10,
        num_workers=4)
    gen.start()

    images, weight_characters, weight_affinitys, normal_images = gen.get_batch()
    label = np.stack((weight_characters, weight_affinitys), axis=-1)
    for i in range(len(images)):
        print('Saving images')
        img = denormalize_mean_variance(images[i])[:,:,::-1]
        cv2.imwrite("tmp/img_{}.png".format(i), img)
        cv2.imwrite("tmp/normal_{}.png".format(i), normal_images[i])
        cv2.imwrite("tmp/character_{}.png".format(i), (255 * weight_characters[i]).astype(np.uint8))
        cv2.imwrite("tmp/aff_{}.png".format(i), (255 * weight_affinitys[i]).astype(np.uint8))

    from utils.utils import visual_heatmap
    img = denormalize_mean_variance(images[0])[:,:,::-1]

    w_c = weight_characters[0]
    w_a = weight_affinitys[0]
    image = visual_heatmap(img, w_c)
    cv2.imwrite('heat_c.png', (255 * w_c).astype(np.uint8))
    image = visual_heatmap(img, w_a)
    cv2.imwrite('heat_a.png', (255 * w_a).astype(np.uint8))

    gen.kill()
    del gen
    