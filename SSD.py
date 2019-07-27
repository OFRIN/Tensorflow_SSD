
import numpy as np
import tensorflow as tf

import VGG16 as vgg
from Define import *

initializer = tf.contrib.layers.xavier_initializer()

def conv_bn_relu(x, filters, kernel_size, strides, padding, is_training, scope, bn = True, activation = True):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d(inputs = x, filters = filters, kernel_size = kernel_size, strides = strides, padding = padding, kernel_initializer = initializer, name = 'conv2d')
        
        if bn:
            x = tf.layers.batch_normalization(inputs = x, training = is_training, name = 'bn')

        if activation:
            x = tf.nn.relu(x, name = 'relu')
    return x

VGG_MEAN = [103.94, 116.78, 123.68]
def SSD_VGG(x, is_training):
    x -= VGG_MEAN
    feature_maps_list = []
    
    with tf.contrib.slim.arg_scope(vgg.vgg_arg_scope()):
        conv4_3, conv5_3 = vgg.vgg_16(x, num_classes=1000, is_training=is_training, dropout_keep_prob=0.5)

    conv4_3 = tf.nn.l2_normalize(conv4_3, axis = -1)
    conv4_3 = conv4_3 * tf.Variable(initial_value = tf.constant(20.), name = 'norm_factor')
    print(conv4_3); feature_maps_list.append(conv4_3)

    with tf.variable_scope('SSD'):
        conv6 = conv_bn_relu(conv5_3, 1024, (3, 3), 1, 'same', is_training, 'conv6')
        conv7 = conv_bn_relu(conv6, 1024, (1, 1), 1, 'same', is_training, 'conv7')
        print(conv7); feature_maps_list.append(conv7)

        conv8_1 = conv_bn_relu(conv7, 256, (1, 1), 1, 'same', is_training, 'conv8_1')
        conv8_2 = conv_bn_relu(conv8_1, 512, (3, 3), 2, 'same', is_training, 'conv8_2')
        print(conv8_2); feature_maps_list.append(conv8_2)
        
        conv9_1 = conv_bn_relu(conv8_2, 128, (1, 1), 1, 'same', is_training, 'conv9_1')
        conv9_2 = conv_bn_relu(conv9_1, 256, (3, 3), 2, 'same', is_training, 'conv9_2')
        print(conv9_2); feature_maps_list.append(conv9_2)

        conv10_1 = conv_bn_relu(conv9_2, 128, (1, 1), 1, 'same', is_training, 'conv10_1')
        conv10_2 = conv_bn_relu(conv10_1, 256, (3, 3), 2, 'same', is_training, 'conv10_2')
        print(conv10_2); feature_maps_list.append(conv10_2)

        conv11_1 = conv_bn_relu(conv10_2, 128, (1, 1), 1, 'same', is_training, 'conv11_1')
        conv11_2 = conv_bn_relu(conv11_1, 256, (3, 3), 1, 'valid', is_training, 'conv11_2')
        print(conv11_2); feature_maps_list.append(conv11_2)

    # SSD_Classifiers
    with tf.variable_scope('SSD_Classifiers'):
        
        pred_bboxes_list = []
        pred_classes_list = []

        for index, feature_maps in enumerate(feature_maps_list):
            print('# ', feature_maps)
            feature_h, feature_w, feature_c = feature_maps.shape[1:]
            bbox_count = len(SSD_ASPECT_RATIOS[index])

            # bboxes
            pred_bboxes = conv_bn_relu(feature_maps, 4 * bbox_count, (1, 1), 1, 'same', is_training, 'BoundingBoxes_{}'.format(index), True, False)
            pred_bboxes = tf.reshape(pred_bboxes, [-1, feature_w * feature_h * bbox_count, 4], name = 'BoundingBoxes_{}_Reshape'.format(index))

            # class
            pred_classes = conv_bn_relu(feature_maps, CLASSES * bbox_count, (1, 1), 1, 'same', is_training, 'Classes_{}'.format(index), True, False)
            pred_classes = tf.reshape(pred_classes, [-1, feature_w * feature_h * bbox_count, CLASSES], name = 'Classes_{}_Reshape'.format(index))

            pred_bboxes_list.append(pred_bboxes)
            pred_classes_list.append(pred_classes)

        pred_bboxes_op = tf.concat(pred_bboxes_list, axis = 1, name = 'offset_bboxes')
        pred_classes_op = tf.concat(pred_classes_list, axis = 1, name = 'classes')
        
    return pred_bboxes_op, pred_classes_op

if __name__ == '__main__':
    input_var = tf.placeholder(tf.float32, [None, 300, 300, 3])
    
    pred_bboxes_op, pred_classes_op = SSD_VGG(input_var, False)
    print(pred_bboxes_op, pred_classes_op)
    