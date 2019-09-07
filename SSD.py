# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import numpy as np
import tensorflow as tf

import resnet_v2.resnet_v2 as resnet_v2

from Define import *

initializer = tf.contrib.layers.xavier_initializer()

def conv_bn_relu(x, filters, kernel_size, strides, padding, is_training, scope, bn = True, activation = True, use_bias = True, upscaling = False):
    with tf.variable_scope(scope):
        if not upscaling:
            x = tf.layers.conv2d(inputs = x, filters = filters, kernel_size = kernel_size, strides = strides, padding = padding, kernel_initializer = initializer, use_bias = use_bias, name = 'conv2d')
        else:
            x = tf.layers.conv2d_transpose(inputs = x, filters = filters, kernel_size = kernel_size, strides = strides, padding = padding, kernel_initializer = initializer, use_bias = use_bias, name = 'upconv2d')
        
        if bn:
            x = tf.layers.batch_normalization(inputs = x, training = is_training, name = 'bn')

        if activation:
            x = tf.nn.relu(x, name = 'relu')
    return x

def SSD_Decode_Layer(offset_bboxes, anchors):
    with tf.variable_scope('SSD_Decode'):
        # 1. offset_bboxes
        tx = offset_bboxes[..., 0]
        ty = offset_bboxes[..., 1]
        tw = offset_bboxes[..., 2]
        th = offset_bboxes[..., 3]
        
        # 2. anchors
        wa = anchors[..., 2] - anchors[..., 0]
        ha = anchors[..., 3] - anchors[..., 1]
        xa = anchors[..., 0] + wa / 2
        ya = anchors[..., 1] + ha / 2

        # 3. pred_bboxes (cxcywh)
        x = tx * wa + xa
        y = ty * ha + ya
        w = tf.exp(tw) * wa
        h = tf.exp(th) * ha
        
        # 4. pred_bboxes (cxcywh -> xyxy)
        xmin = x - w / 2
        ymin = y - h / 2
        xmax = x + w / 2
        ymax = y + h / 2

        # 5. exception (0 ~ IMAGE_WIDTH , IMAGE_HEIGHT)
        xmin = tf.clip_by_value(xmin[..., tf.newaxis], 0, IMAGE_WIDTH - 1)
        ymin = tf.clip_by_value(ymin[..., tf.newaxis], 0, IMAGE_HEIGHT - 1)
        xmax = tf.clip_by_value(xmax[..., tf.newaxis], 0, IMAGE_WIDTH - 1)
        ymax = tf.clip_by_value(ymax[..., tf.newaxis], 0, IMAGE_HEIGHT - 1)
        
        pred_bboxes = tf.concat([xmin, ymin, xmax, ymax], axis = -1)

    return pred_bboxes

def SSD_ResNet_50(input_var, is_training):
    ssd_dic = {}
    ssd_sizes = []

    x = input_var - [103.939, 123.68, 116.779]
    
    with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits, end_points = resnet_v2.resnet_v2_50(x, is_training = is_training)

    feature_maps = [end_points['resnet_v2_50/block{}'.format(i)] for i in [1, 2, 4]]
    
    with tf.variable_scope('SSD'):
        x = feature_maps[2]

        x = conv_bn_relu(x, 128, (1, 1), 1, 'valid', is_training, 'conv1_1x1')
        x = conv_bn_relu(x, 256, (3, 3), 2, 'same', is_training, 'conv1_3x3')
        feature_maps.append(x)
        
        x = conv_bn_relu(x, 128, (1, 1), 1, 'valid', is_training, 'conv2_1x1')
        x = conv_bn_relu(x, 256, (3, 3), 2, 'same', is_training, 'conv2_3x3')
        feature_maps.append(x)

        x = conv_bn_relu(x, 128, (1, 1), 1, 'valid', is_training, 'conv3_1x1')
        x = conv_bn_relu(x, 256, (3, 3), 1, 'valid', is_training, 'conv3_3x3')
        feature_maps.append(x)
        
        '''
        Tensor("resnet_v2_50/block1/unit_3/bottleneck_v2/add:0", shape=(?, 41, 41, 256), dtype=float32)
        Tensor("resnet_v2_50/block2/unit_4/bottleneck_v2/add:0", shape=(?, 21, 21, 512), dtype=float32)
        Tensor("resnet_v2_50/block4/unit_3/bottleneck_v2/add:0", shape=(?, 11, 11, 2048), dtype=float32)
        Tensor("SSD/conv1_3x3/relu:0", shape=(?, 6, 6, 128), dtype=float32)
        Tensor("SSD/conv2_3x3/relu:0", shape=(?, 3, 3, 128), dtype=float32)
        Tensor("SSD/conv3_3x3/relu:0", shape=(?, 1, 1, 128), dtype=float32)
        Tensor("SSD/bboxes:0", shape=(?, 13734, 4), dtype=float32)
        Tensor("SSD/Softmax:0", shape=(?, 13734, 21), dtype=float32)
        '''
        # for feature_map in feature_maps:
        #    print(feature_map)

        pred_bboxes = []
        pred_classes = []

        anchors_per_location = len(ANCHOR_SCALES) * len(ANCHOR_RATIOS)
        
        for feature_map in feature_maps:
            _, h, w, c = feature_map.shape.as_list()
            ssd_sizes.append([w, h])
            
            _pred_bboxes = conv_bn_relu(feature_map, 256, (3, 3), 1, 'same', is_training, 'bboxes_{}x{}_1'.format(w, h), bn = True, activation = True)
            _pred_bboxes = conv_bn_relu(_pred_bboxes, 256, (3, 3), 1, 'same', is_training, 'bboxes_{}x{}_2'.format(w, h), bn = True, activation = True)
            _pred_bboxes = conv_bn_relu(_pred_bboxes, 4 * anchors_per_location, (3, 3), 1, 'same', is_training, 'bboxes_{}x{}'.format(w, h), bn = False, activation = False)
            _pred_bboxes = tf.reshape(_pred_bboxes, [-1, h * w * anchors_per_location, 4])
            
            _pred_classes = conv_bn_relu(feature_map, 256, (3, 3), 1, 'same', is_training, 'classes_{}x{}_1'.format(w, h), bn = True, activation = True)
            _pred_classes = conv_bn_relu(_pred_classes, 256, (3, 3), 1, 'same', is_training, 'classes_{}x{}_2'.format(w, h), bn = True, activation = True)
            _pred_classes = conv_bn_relu(_pred_classes, CLASSES * anchors_per_location, (3, 3), 1, 'same', is_training, 'classes_{}x{}'.format(w, h), bn = False, activation = False)
            _pred_classes = tf.reshape(_pred_classes, [-1, h * w* anchors_per_location, CLASSES])

            pred_bboxes.append(_pred_bboxes)
            pred_classes.append(_pred_classes)

        pred_bboxes = tf.concat(pred_bboxes, axis = 1, name = 'bboxes')
        pred_classes = tf.concat(pred_classes, axis = 1, name = 'classes')

        ssd_dic['pred_bboxes'] = pred_bboxes
        ssd_dic['pred_classes'] = tf.nn.softmax(pred_classes, axis = -1)

    return ssd_dic, ssd_sizes

SSD = SSD_ResNet_50

if __name__ == '__main__':
    input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])

    ssd_dic, ssd_sizes = SSD(input_var, False)
    
    print(ssd_dic['pred_bboxes'])
    print(ssd_dic['pred_classes'])
