import tensorflow as tf

from Define import *

def Smooth_L1(x):
    return tf.where(tf.less_equal(tf.abs(x), 1.0), tf.multiply(0.5, tf.pow(x, 2.0)), tf.subtract(tf.abs(x), 0.5))

def SSD_Loss(pred_offset_bboxes, pred_classes, gt_offset_bboxes, gt_classes, gt_positives, length):
    # 1. get num
    total_num = tf.ones([BATCH_SIZE], dtype = tf.float32) * tf.to_float(length)
    
    positive_num = tf.reduce_sum(gt_positives, axis = -1)
    negative_num = total_num - positive_num

    # 2. get mask
    positive_masks = gt_positives
    negative_masks = 1 - gt_positives # [BATCH_SIZE, length]

    top_k_negative_num = tf.cast(tf.minimum(negative_num, positive_num * 3), tf.int32)
    background_probs = tf.where(tf.cast(negative_masks, dtype = tf.bool), tf.nn.softmax(pred_classes, axis = -1)[:, :, 0], tf.ones_like(pred_classes[:, :, 0])) # [BATCH_SIZE, length]
    
    top_k_negative_masks = []
    for i in range(BATCH_SIZE):
        top_values, top_indexs = tf.nn.top_k(-background_probs[i], k = top_k_negative_num[i])

        cond = background_probs[i] < -top_values[-1]
        top_k_negative_mask = negative_masks * tf.where(cond, tf.ones_like(background_probs[i]), tf.zeros_like(background_probs[i]))
        top_k_negative_masks.append(top_k_negative_mask)

    top_k_negative_masks = tf.convert_to_tensor(top_k_negative_masks, dtype = tf.float32, name = 'top_k_negative_masks')
    
    # get classification losses
    class_loss = tf.nn.softmax_cross_entropy_with_logits(logits = pred_classes, labels = gt_classes) # return (BATCH_SIZE, length)
    
    positive_class_loss = positive_masks * class_loss # return (BATCH_SIZE, length) - positive = value, negative = 0
    positive_class_loss = tf.reduce_sum(positive_class_loss, axis = -1) # return (BATCH_SIZE)

    negative_class_loss = top_k_negative_masks * class_loss # return (BATCH_SIZE, length) - positive = 0, negative = 0, top_k_negative = value
    negative_class_loss = tf.reduce_sum(negative_class_loss, axis = -1)
    
    class_loss = positive_class_loss + negative_class_loss
    class_loss = class_loss / positive_num
    class_loss = tf.reduce_mean(class_loss, name = 'class_loss')
    
    # get localization losses
    location_loss = tf.reduce_sum(Smooth_L1(pred_offset_bboxes - gt_offset_bboxes), axis = -1)

    location_loss = positive_masks * location_loss
    location_loss = tf.reduce_sum(location_loss, axis = -1)

    location_loss = location_loss / positive_num
    location_loss = tf.reduce_mean(location_loss, name = 'location_loss')
    
    # total loss
    return class_loss + location_loss, class_loss, location_loss

