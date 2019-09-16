
# dataset parameters
ROOT_DIR = 'D:/_DeepLearning_DB/'

CLASS_NAMES = ['background'] + ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
CLASS_DIC = {class_name : index for index, class_name in enumerate(CLASS_NAMES)}
CLASSES = len(CLASS_NAMES)

# network parameters
IMAGE_HEIGHT = 300
IMAGE_WIDTH = 300
IMAGE_CHANNEL = 3

ANCHOR_SCALES = [0.5, 1.0]
ANCHOR_RATIOS = [1./3, 1./2, 1.0, 2.0, 3.0]

POSITIVE_IOU_THRESHOLD = 0.5

# loss parameters
WEIGHT_DECAY = 0.0001

# train
NUM_GPU = 2
BATCH_SIZE = 16 * NUM_GPU
INIT_LEARNING_RATE = 1e-4

MAX_EPOCH = 200
LOG_ITERATION = 50
VALID_ITERATION = 10000
