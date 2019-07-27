import os
import cv2
import sys
import glob
import time

import numpy as np
import tensorflow as tf

from SSD import *
from SSD_Loss import *
from SSD_Utils import *

from Define import *
from Utils import *
from DataAugmentation import *

os.environ["CUDA_VISIBLE_DEVICES"]="1"

# 1. dataset
TRAIN_XML_DIRS = ["D:/_ImageDataset/VOC2007/train/xml/", "D:/_ImageDataset/VOC2012/xml/"]
TEST_XML_DIRS = ["D:/_ImageDataset/VOC2007/test/xml/"]

train_xml_paths = []
test_xml_paths = []

for train_xml_dir in TRAIN_XML_DIRS:
    train_xml_paths += glob.glob(train_xml_dir + "*")

for test_xml_dir in TEST_XML_DIRS:
    test_xml_paths += glob.glob(test_xml_dir + "*")

np.random.shuffle(train_xml_paths)
train_xml_paths = np.asarray(train_xml_paths)

valid_xml_paths = train_xml_paths[:int(len(train_xml_paths) * 0.1)]
train_xml_paths = train_xml_paths[int(len(train_xml_paths) * 0.1):]

log_print('[i] Train : {}'.format(len(train_xml_paths)))
log_print('[i] Valid : {}'.format(len(valid_xml_paths)))
log_print('[i] Test : {}'.format(len(test_xml_paths)))

# 2. build
ssd_edcoder = SSD_EDCoder(SSD_LAYER_SHAPES, SSD_ASPECT_RATIOS, [MIN_SCALE, MAX_SCALE], POSITIVE_IOU_THRESHOLD)

input_var = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
is_training = tf.placeholder(tf.bool)

gt_classes_op = tf.placeholder(tf.float32, [BATCH_SIZE, ssd_edcoder.length, CLASSES])
gt_offset_bboxes_op = tf.placeholder(tf.float32, [BATCH_SIZE, ssd_edcoder.length, 4])
gt_positives_op = tf.placeholder(tf.float32, [BATCH_SIZE, ssd_edcoder.length])

learning_rate_var = tf.placeholder(tf.float32)

pred_offset_bboxes_op, pred_classes_op = SSD_VGG(input_var, is_training)

loss_op, class_loss_op, location_loss_op = SSD_Loss(pred_offset_bboxes_op, pred_classes_op, gt_offset_bboxes_op, gt_classes_op, gt_positives_op, ssd_edcoder.length)

vars = tf.trainable_variables()
l2_reg_loss_op = tf.add_n([tf.nn.l2_loss(var) for var in vars]) * WEIGHT_DECAY
loss_op += l2_reg_loss_op

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    #train_op = tf.train.MomentumOptimizer(learning_rate_var, momentum = 0.9).minimize(loss_op)    
    train_op = tf.train.AdamOptimizer(learning_rate_var).minimize(loss_op)

# 3. train
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#'''
vgg_vars = []
for var in vars:
    if 'vgg_16' in var.name:
        vgg_vars.append(var)

restore_saver = tf.train.Saver(var_list = vgg_vars)
restore_saver.restore(sess, './vgg_16/vgg_16.ckpt')
log_print('[i] restored imagenet parameters (VGG16)')
#'''

saver = tf.train.Saver()

restore_index = 0
if restore_index != 0:
    saver.restore(sess, './model/SSD_VGG_{}.ckpt'.format(restore_index))

batch_indexs = np.arange(len(train_xml_paths), dtype = np.int32)
batch_image_data = np.zeros([BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL], dtype = np.float32)

loss_list = []
class_loss_list = []
location_loss_list = []
l2_reg_loss_list = []

best_valid_mAP = 0.0

st = time.time()

valid_length = len(valid_xml_paths)
valid_iterations = valid_length // BATCH_SIZE

learning_rate = INIT_LEARNING_RATE

train_save_dir = './debug/train/'
if not os.path.isdir(train_save_dir):
    os.makedirs(train_save_dir)

valid_save_dir = './debug/valid/'
if not os.path.isdir(valid_save_dir):
    os.makedirs(valid_save_dir)

for iter in range(restore_index + 1, MAX_ITERATIONS + 1):
    if iter % 20000 == 0:
        learning_rate /= 10
        log_print('[i] learning rate decay : {} -> {}'.format(learning_rate * 10, learning_rate))

    np.random.shuffle(batch_indexs)

    batch_data_list = []
    batch_xml_paths = train_xml_paths[batch_indexs[:BATCH_SIZE]]
    batch_image_data = np.zeros([BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL], dtype = np.float32)
    
    for i, xml_path in enumerate(batch_xml_paths):
        image_path, bboxes, classes = xml_read(xml_path, normalize = False)

        image = cv2.imread(image_path)

        bboxes = np.asarray(bboxes).astype(np.int32)
        classes = np.asarray(classes).astype(np.int32)

        image, bboxes = random_flip(image, bboxes)
        image, bboxes = random_scale(image, bboxes)
        image = random_blur(image)
        image = random_brightness(image)
        image = random_hue(image)
        image = random_saturation(image)
        image = random_gray(image)
        image, bboxes, classes = random_shift(image, bboxes, classes)
        image, bboxes, classes = random_crop(image, bboxes, classes)
        image, bboxes, classes = random_translate(image, bboxes, classes)

        h, w, c = image.shape
        bboxes = bboxes.astype(np.float32)
        bboxes /= [w, h, w, h]

        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        batch_image_data[i] = image.astype(np.float32).copy()
        
        data = []
        for bbox, class_index in zip(bboxes, classes):
            
            xmin, ymin, xmax, ymax = bbox
            if (xmax - xmin) * (ymax - ymin) > 0.0:
                data.append([bbox, class_index])
            else:
                print(bbox)
                print(xml_path)
                input()

        batch_data_list.append(data)
    
    batch_gt_classes, batch_gt_offset_bboxes, batch_gt_positives = ssd_edcoder.Encode(batch_data_list)
    
    _feed_dict = {input_var : batch_image_data, is_training : True, learning_rate_var : learning_rate,
                  gt_classes_op : batch_gt_classes, gt_offset_bboxes_op : batch_gt_offset_bboxes, gt_positives_op : batch_gt_positives}
    _, loss, class_loss, location_loss, l2_reg_loss = sess.run([train_op, loss_op, class_loss_op, location_loss_op, l2_reg_loss_op], feed_dict = _feed_dict)

    loss_list.append(loss)
    class_loss_list.append(class_loss)
    location_loss_list.append(location_loss)
    l2_reg_loss_list.append(l2_reg_loss)

    assert not np.isnan(loss), "[!] Loss = Nan"

    if iter % LOG_ITERATIONS == 0:
        train_time = int(time.time() - st)

        loss = np.mean(loss_list)
        class_loss = np.mean(class_loss_list)
        location_loss = np.mean(location_loss_list)
        l2_reg_loss = np.mean(l2_reg_loss_list)
        
        log_print('[i] iter : {}, loss : {:.4f}, class_loss : {:.4f}, location_loss : {:.4f}, l2_reg_loss : {:.4f}, time : {}sec'.format(iter, loss, class_loss, location_loss, l2_reg_loss, train_time))

        st = time.time()

        loss_list = []
        class_loss_list = []
        location_loss_list = []
        l2_reg_loss_list = []
    
    if iter % TRAIN_ITERATIONS == 0:
        # train set
        precision_list = []
        recall_list = []

        save_count = 0

        for i in range(TRAIN_TEST_ITERATIONS):
            sys.stdout.write('\r[{}/{}]'.format(i, TRAIN_TEST_ITERATIONS))
            sys.stdout.flush()

            batch_xml_paths = train_xml_paths[batch_indexs[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]]

            batch_image_data = []
            batch_train_image_data = np.zeros((BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL), dtype = np.float32)
            batch_gt_bboxes = []
            batch_gt_classes = []
            batch_sizes = []

            for index, xml_path in enumerate(batch_xml_paths):
                image_path, gt_bboxes, gt_classes = xml_read(xml_path)

                image = cv2.imread(image_path)
                h, w, c = image.shape

                batch_image_data.append(image)
                batch_train_image_data[index] = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT)).astype(np.float32)
                batch_gt_bboxes.append(gt_bboxes)
                batch_gt_classes.append(gt_classes)
                batch_sizes.append([w, h])

            batch_pred_classes, batch_pred_offset_bboxes = sess.run([pred_classes_op, pred_offset_bboxes_op], feed_dict = {input_var : batch_train_image_data, is_training : False})

            for index in range(len(batch_xml_paths)):
                w, h = batch_sizes[index]

                gt_bboxes, gt_classes = batch_gt_bboxes[index], batch_gt_classes[index]
                pred_bboxes, pred_classes = ssd_edcoder.Decode(batch_pred_classes[index], batch_pred_offset_bboxes[index], size = (w, h))
                pred_bboxes, pred_classes = class_nms(pred_bboxes, pred_classes)

                if save_count < 5:
                    save_count += 1
                    
                    sample_image = batch_image_data[index]
                    for pred_bbox, pred_class in zip(pred_bboxes, pred_classes):
                        xmin, ymin, xmax, ymax, _ = pred_bbox.astype(np.int32)

                        cv2.putText(sample_image, CLASS_NAMES[pred_class], (xmin, ymin - 10), 1, 1, (0, 255, 0), 2)
                        cv2.rectangle(sample_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.imwrite(train_save_dir + 'iter={}_count={}.jpg'.format(iter, save_count), sample_image)

                precision, recall = Precision_Recall(gt_bboxes, gt_classes, pred_bboxes, pred_classes)

                precision_list.append(precision)
                recall_list.append(recall)
        
        precision = np.mean(precision_list) * 100
        recall = np.mean(recall_list) * 100
        mAP = (precision + recall) / 2

        log_print('[i] train mAP : {:.2f}%, precision : {:.2f}%, recall : {:.2f}%'.format(mAP, precision, recall))

    if iter % VALID_ITERATIONS == 0:
        
        # validation set
        precision_list = []
        recall_list = []

        save_count = 0

        for i in range(valid_iterations):
            sys.stdout.write('\r[{}/{}]'.format(i, valid_iterations))
            sys.stdout.flush()

            batch_xml_paths = valid_xml_paths[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]

            batch_image_data = []
            batch_train_image_data = np.zeros((BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL), dtype = np.float32)
            batch_gt_bboxes = []
            batch_gt_classes = []
            batch_sizes = []

            for index, xml_path in enumerate(batch_xml_paths):
                image_path, gt_bboxes, gt_classes = xml_read(xml_path)

                image = cv2.imread(image_path)
                h, w, c = image.shape

                batch_image_data.append(image)
                batch_train_image_data[index] = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT)).astype(np.float32)
                batch_gt_bboxes.append(gt_bboxes)
                batch_gt_classes.append(gt_classes)
                batch_sizes.append([w, h])

            batch_pred_classes, batch_pred_offset_bboxes = sess.run([pred_classes_op, pred_offset_bboxes_op], feed_dict = {input_var : batch_train_image_data, is_training : False})

            for index in range(len(batch_xml_paths)):
                w, h = batch_sizes[index]

                gt_bboxes, gt_classes = batch_gt_bboxes[index], batch_gt_classes[index]
                pred_bboxes, pred_classes = ssd_edcoder.Decode(batch_pred_classes[index], batch_pred_offset_bboxes[index], size = (w, h))
                pred_bboxes, pred_classes = class_nms(pred_bboxes, pred_classes)

                if save_count < 5:
                    save_count += 1
                    
                    sample_image = batch_image_data[index]
                    for pred_bbox, pred_class in zip(pred_bboxes, pred_classes):
                        xmin, ymin, xmax, ymax, _ = pred_bbox.astype(np.int32)

                        cv2.putText(sample_image, CLASS_NAMES[pred_class], (xmin, ymin - 10), 1, 1, (0, 255, 0), 2)
                        cv2.rectangle(sample_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.imwrite(valid_save_dir + 'iter={}_count={}.jpg'.format(iter, save_count), sample_image)

                precision, recall = Precision_Recall(gt_bboxes, gt_classes, pred_bboxes, pred_classes)

                precision_list.append(precision)
                recall_list.append(recall)

        precision = np.mean(precision_list) * 100
        recall = np.mean(recall_list) * 100
        mAP = (precision + recall) / 2
        print()

        if best_valid_mAP < mAP:
            best_valid_mAP = mAP
            saver.save(sess, './model/SSD_VGG_{}.ckpt'.format(iter))

            log_print('[i] precision : {:.2f}%'.format(precision))
            log_print('[i] recall : {:.2f}%'.format(recall))

        log_print('[i] valid mAP : {:.2f}%, best valid mAP : {:.2f}%'.format(mAP, best_valid_mAP))
