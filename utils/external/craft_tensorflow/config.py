import os
from shutil import rmtree
import math

seed = 0

# ====== data_manipulation =====
window = 120
sigma = 18.5
sigma_aff = 20
threshold_point = 25
THRESHOLD_POSITIVE = 0.1
# ====== data_manipulation =====

# ====== Loss hard negative mining =====
threshold_positive = 0.1
threshold_negative = 0.1
# ====== Loss hard negative mining =====

# ====== training agrs =================
batch_size = 4
num_epochs = 1000
learning_rate = 0.001
decay_factor_lr = 0.0125/4000

train_mat = '/hdd/UBD/background-images/data_generated/15_11/train/synth/bg.mat'
prefix_path_train = '/hdd/UBD/background-images/data_generated/15_11/train/synth/img/'

val_mat = '/hdd/UBD/background-images/data_generated/15_11/val/synth/bg.mat'
prefix_path_val = '/hdd/UBD/background-images/data_generated/15_11/val/synth/img/'

path_save_model = 'checkpoints'
interval_save = 10000 # 10000 iteration save
pretraing_model_path = 'pretrained_model/model_0000170000.ckpt' # None or not exist, we will train from scratch
# ====== training agrs =================

# ====== ICDAR15 =================
train_weakly = True
prob_weakly = 1/6

ICDAR15_img_dir = '/home/thanhtm/Desktop/icdar2015/train_images/'
ICDAR15_lbl_dir = '/home/thanhtm/Desktop/icdar2015/train_gts/'
DONTCARE_LABEL = list(['###'])
# ====== ICDAR15 =================

# ====== validate ================
boundary_character = math.exp(-1/2*(threshold_point**2)/(sigma**2))
boundary_affinity = math.exp(-1/2*(threshold_point**2)/(sigma_aff**2))
threshold_character = boundary_character + 0.03
threshold_affinity = boundary_affinity + 0.03

threshold_character_upper = boundary_character + 0.2
threshold_affinity_upper = boundary_affinity + 0.2
scale_character = math.sqrt(math.log(boundary_character)/math.log(threshold_character_upper))
scale_affinity = math.sqrt(math.log(boundary_affinity)/math.log(threshold_affinity_upper))

threshold_word = 0.7
# ====== validate ================

debug_mode = False
DEBUG_DIR = 'debug_img'
if os.path.exists(DEBUG_DIR):
    rmtree(DEBUG_DIR)
os.makedirs(DEBUG_DIR)

