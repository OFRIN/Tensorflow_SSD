# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import os
import cv2
import sys
import glob
import time
import random

import numpy as np
import tensorflow as tf

from Define import *
from Utils import *

from SSD import *
from SSD_Utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 1. dataset
test_xml_paths = glob.glob(ROOT_DIR + 'VOC2007/test/xml/*.xml')
print('[i] Test : {}'.format(len(test_xml_paths)))

# 2. build
input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
is_training = tf.placeholder(tf.bool)

ssd_dic, ssd_sizes = SSD(input_var, is_training)
anchors = generate_anchors(ssd_sizes, [IMAGE_WIDTH, IMAGE_HEIGHT], ANCHOR_SCALES, ANCHOR_RATIOS)

pred_bboxes_op = SSD_Decode_Layer(ssd_dic['pred_bboxes'], anchors)
pred_classes_op = ssd_dic['pred_classes']

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, './model/SSD_ResNet_v2_{}.ckpt'.format(60000))

test_iteration = len(test_xml_paths) // BATCH_SIZE

precision_list = []
recall_list = []

for test_iter in range(test_iteration):
    sizes = []
    batch_image_data = []
    batch_gt_bboxes = []
    batch_gt_classes = []

    batch_xml_paths = test_xml_paths[test_iter * BATCH_SIZE : (test_iter + 1) * BATCH_SIZE]
    for xml_path in batch_xml_paths:
        image, gt_bboxes, gt_classes = get_data(xml_path, training = False, normalize = False)
        h, w, c = image.shape

        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)
        
        sizes.append([w, h])
        batch_image_data.append(image)
        batch_gt_bboxes.append(gt_bboxes)
        batch_gt_classes.append(gt_classes)

    batch_image_data = np.asarray(batch_image_data, dtype = np.float32)
    batch_encode_bboxes, batch_encode_classes = sess.run([pred_bboxes_op, pred_classes_op], feed_dict = {input_var : batch_image_data, is_training : False})

    for encode_bboxes, encode_classes, gt_bboxes, gt_classes, size in zip(batch_encode_bboxes, batch_encode_classes, batch_gt_bboxes, batch_gt_classes, sizes):
        pred_bboxes, pred_classes = Decode(encode_bboxes, encode_classes, anchors, size)

        precision, recall = Precision_Recall(gt_bboxes, gt_classes, pred_bboxes, pred_classes)
        
        precision_list.append(precision)
        recall_list.append(recall)

precision = np.mean(precision_list) * 100
recall = np.mean(recall_list) * 100
mAP = (precision + recall) / 2

print('[i] precision : {:.2f}%'.format(precision))
print('[i] recall : {:.2f}%'.format(recall))
print('[i] mAP : {:.2f}'.format(mAP))

'''
[i] precision : 84.02%
[i] recall : 73.29%
[i] mAP : 78.65
'''