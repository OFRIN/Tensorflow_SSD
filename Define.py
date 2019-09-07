# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

# dataset parameters
ROOT_DIR = 'D:/_DeepLearning_DB/'

CLASS_NAMES = ['background'] + ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
CLASS_DIC = {class_name : index for index, class_name in enumerate(CLASS_NAMES)}
CLASSES = len(CLASS_NAMES)

VALID_SET_RATIO = 0.1

# network parameters
IMAGE_HEIGHT = 321
IMAGE_WIDTH = 321
IMAGE_CHANNEL = 3

SCALE_FACTORS = [0.1, 0.1, 0.2, 0.2]

ANCHOR_SCALES = [1.0, 2.0]
ANCHOR_RATIOS = [1./2, 1./3, 1.0, 2.0, 3.]

POSITIVE_IOU_THRESHOLD = 0.5

# loss parameters
WEIGHT_DECAY = 0.0001

# train
BATCH_SIZE = 32
INIT_LEARNING_RATE = 1e-4

MAX_EPOCH = 100
LOG_ITERATION = 50
VALID_ITERATION = 5000