
IMAGE_WIDTH = 300
IMAGE_HEIGHT = 300
IMAGE_CHANNEL = 3

CLASS_NAMES = ['background', 
               'aeroplane','bicycle','bird','boat','bottle',
               'bus','car','cat','chair','cow',
               'diningtable','dog','horse','motorbike','person',
               'pottedplant','sheep','sofa','train','tvmonitor']

CLASS_DIC = {}
for value, key in enumerate(CLASS_NAMES):
    CLASS_DIC[key] = value

CLASSES = len(CLASS_NAMES)
CLASSIFIER_SIZE = 4 + CLASSES

BATCH_SIZE = 32
INIT_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0005

# ~ 60k : 1e-3
# ~ 20k : 1e-4
MAX_ITERATIONS = 80000 # 80k
LOG_ITERATIONS = 10
TRAIN_ITERATIONS = 100
VALID_ITERATIONS = 1000

TRAIN_TEST_ITERATIONS = 1

# size, bbox scale, bbox count, bbox aspect ratio
SSD_LAYER_SHAPES = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
SSD_ASPECT_RATIOS = [[1, 2, 1/2, 1/3], 
                     [1, 2, 3, 1/2, 1/3, -1], 
                     [1, 2, 3, 1/2, 1/3, -1], 
                     [1, 2, 3, 1/2, 1/3, -1],
                     [1, 2, 1/2, 1/3],
                     [1, 2, 1/2, 1/3]]

MIN_SCALE = 0.1
MAX_SCALE = 0.9
POSITIVE_IOU_THRESHOLD = 0.5
