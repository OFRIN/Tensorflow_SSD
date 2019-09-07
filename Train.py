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
from Teacher import *
from DataAugmentation import *

from SSD import *
from SSD_Loss import *
from SSD_Utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 1. dataset
train_xml_paths = glob.glob(ROOT_DIR + 'VOC2007/train/xml/*.xml') + glob.glob(ROOT_DIR + 'VOC2012/xml/*.xml')
valid_xml_paths = random.sample(train_xml_paths, int(len(train_xml_paths) * VALID_SET_RATIO))

train_xml_paths = list(set(train_xml_paths) - set(valid_xml_paths))

open('log.txt', 'w')
log_print('[i] Train : {}'.format(len(train_xml_paths)))
log_print('[i] Valid : {}'.format(len(valid_xml_paths)))

# 2. build
input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
is_training = tf.placeholder(tf.bool)

ssd_dic, ssd_sizes = SSD(input_var, is_training)
anchors = generate_anchors(ssd_sizes, [IMAGE_WIDTH, IMAGE_HEIGHT], ANCHOR_SCALES, ANCHOR_RATIOS)

gt_bboxes_var = tf.placeholder(tf.float32, [None, anchors.shape[0], 4])
gt_classes_var = tf.placeholder(tf.float32, [None, anchors.shape[0], CLASSES])

pred_bboxes_op = SSD_Decode_Layer(ssd_dic['pred_bboxes'], anchors)
pred_classes_op = ssd_dic['pred_classes']

log_print('[i] pred_bboxes_op : {}'.format(pred_bboxes_op))
log_print('[i] pred_classes_op : {}'.format(pred_classes_op))
log_print('[i] gt_bboxes_var : {}'.format(gt_bboxes_var))
log_print('[i] gt_classes_var : {}'.format(gt_classes_var))

loss_op, focal_loss_op, giou_loss_op = SSD_Loss(pred_bboxes_op, pred_classes_op, gt_bboxes_var, gt_classes_var)

vars = tf.trainable_variables()
l2_reg_loss_op = tf.add_n([tf.nn.l2_loss(var) for var in vars]) * WEIGHT_DECAY
loss_op += l2_reg_loss_op

tf.summary.scalar('loss', loss_op)
tf.summary.scalar('focal_loss', focal_loss_op)
tf.summary.scalar('giou_loss', giou_loss_op)
tf.summary.scalar('l2_regularization_Loss', l2_reg_loss_op)
summary_op = tf.summary.merge_all()

learning_rate_var = tf.placeholder(tf.float32)
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    # train_op = tf.train.MomentumOptimizer(learning_rate_var, momentum = 0.9).minimize(loss_op)
    train_op = tf.train.AdamOptimizer(learning_rate_var).minimize(loss_op)

# 3. train
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# '''
pretrained_vars = []
for var in vars:
    if 'resnet_v2_50' in var.name:
        pretrained_vars.append(var)

pretrained_saver = tf.train.Saver(var_list = pretrained_vars)
pretrained_saver.restore(sess, './resnet_v2_model/resnet_v2_50.ckpt')
# '''

saver = tf.train.Saver()
# saver.restore(sess, './model/SSD_ResNet_v2_{}.ckpt'.format(10000))

best_valid_mAP = 0.0
learning_rate = INIT_LEARNING_RATE

train_iteration = len(train_xml_paths) // BATCH_SIZE
valid_iteration = len(valid_xml_paths) // BATCH_SIZE

max_iteration = train_iteration * MAX_EPOCH
decay_iteration = np.asarray([0.5 * max_iteration, 0.75 * max_iteration], dtype = np.int32)

log_print('[i] max_iteration : {}'.format(max_iteration))
log_print('[i] decay_iteration : {}'.format(decay_iteration))

loss_list = []
focal_loss_list = []
giou_loss_list = []
l2_reg_loss_list = []
train_time = time.time()

train_writer = tf.summary.FileWriter('./logs/train')

# train_threads = []
# for i in range(2):
#     train_thread = Teacher(train_xml_paths, anchors, max_data_size = 10, debug = True)
#     train_thread.start()
#     train_threads.append(train_thread)

for iter in range(1, max_iteration + 1):
    if iter in decay_iteration:
        learning_rate /= 10
        log_print('[i] learning rate decay : {} -> {}'.format(learning_rate * 10, learning_rate))

    # Teacher (Thread)
    # find = False
    # while not find:
    #     for train_thread in train_threads:
    #         if train_thread.ready:
    #             find = True
    #             batch_image_data, batch_gt_bboxes, batch_gt_classes = train_thread.get_batch_data()        
    #             break

    # Default
    batch_image_data = []
    batch_gt_bboxes = []
    batch_gt_classes = []
    batch_xml_paths = random.sample(train_xml_paths, BATCH_SIZE)

    for xml_path in batch_xml_paths:
        # delay = time.time()

        image, gt_bboxes, gt_classes = get_data(xml_path, training = True)
        
        # image = image.astype(np.uint8)
        # for bbox in gt_bboxes:
        #     xmin, ymin, xmax, ymax = bbox
        #     cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # cv2.imshow('show', image)
        # cv2.waitKey(0)

        # delay = time.time() - delay
        # print('[D] {} = {}ms'.format('xml', int(delay * 1000))) # ~ 41ms

        encode_bboxes, encode_classes = Encode(gt_bboxes, gt_classes, anchors)

        batch_image_data.append(image.astype(np.float32))
        batch_gt_bboxes.append(encode_bboxes)
        batch_gt_classes.append(encode_classes)

    batch_image_data = np.asarray(batch_image_data, dtype = np.float32) 
    batch_gt_bboxes = np.asarray(batch_gt_bboxes, dtype = np.float32)
    batch_gt_classes = np.asarray(batch_gt_classes, dtype = np.float32)

    _feed_dict = {input_var : batch_image_data, gt_bboxes_var : batch_gt_bboxes, gt_classes_var : batch_gt_classes, is_training : True, learning_rate_var : learning_rate}
    log = sess.run([train_op, loss_op, focal_loss_op, giou_loss_op, l2_reg_loss_op, summary_op], feed_dict = _feed_dict)
    # print(log[1:-1])
    
    if np.isnan(log[1]):
        print('[!]', log[1:-1])
        input()

    loss_list.append(log[1])
    focal_loss_list.append(log[2])
    giou_loss_list.append(log[3])
    l2_reg_loss_list.append(log[4])
    train_writer.add_summary(log[5], iter)

    if iter % LOG_ITERATION == 0:
        loss = np.mean(loss_list)
        focal_loss = np.mean(focal_loss_list)
        giou_loss = np.mean(giou_loss_list)
        l2_reg_loss = np.mean(l2_reg_loss_list)
        train_time = int(time.time() - train_time)
        
        log_print('[i] iter : {}, loss : {:.4f}, focal_loss : {:.4f}, giou_loss : {:.4f}, l2_reg_loss : {:.4f}, train_time : {}sec'.format(iter, loss, focal_loss, giou_loss, l2_reg_loss, train_time))

        loss_list = []
        focal_loss_list = []
        giou_loss_list = []
        l2_reg_loss_list = []
        train_time = time.time()

    if iter % VALID_ITERATION == 0:
        precision_list = []
        recall_list = []

        for valid_iter in range(valid_iteration):
            sizes = []
            batch_image_data = []
            batch_gt_bboxes = []
            batch_gt_classes = []

            batch_xml_paths = valid_xml_paths[valid_iter * BATCH_SIZE : (valid_iter + 1) * BATCH_SIZE]
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

        if best_valid_mAP < mAP:
            best_valid_mAP = mAP
            saver.save(sess, './model/SSD_ResNet_v2_{}.ckpt'.format(iter))

            log_print('[i] best precision : {:.2f}%'.format(precision))
            log_print('[i] best recall : {:.2f}%'.format(recall))

        log_print('[i] valid mAP : {:.2f}, best valid mAP : {:.2f}%'.format(mAP, best_valid_mAP))

saver.save(sess, './model/SSD_ResNet_v2.ckpt')

