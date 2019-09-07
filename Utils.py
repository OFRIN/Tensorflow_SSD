# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import glob
import numpy as np

import xml.etree.ElementTree as ET

from Define import *

def log_print(string, log_path = './log.txt'):
    print(string)
    
    f = open(log_path, 'a+')
    f.write(string + '\n')
    f.close()

def xml_read(xml_path, find_labels = CLASS_NAMES, normalize = False):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_path = xml_path[:-3] + '*'
    image_path = image_path.replace('/xml', '/image')
    image_path = glob.glob(image_path)[0]

    size = root.find('size')
    image_width = int(size.find('width').text)
    image_height = int(size.find('height').text)
    
    bboxes = []
    classes = []
    
    for obj in root.findall('object'):
        label = obj.find('name').text
        if not label in find_labels:
            continue
            
        bbox = obj.find('bndbox')
        
        bbox_xmin = max(min(int(bbox.find('xmin').text.split('.')[0]), image_width - 1), 0)
        bbox_ymin = max(min(int(bbox.find('ymin').text.split('.')[0]), image_height - 1), 0)
        bbox_xmax = max(min(int(bbox.find('xmax').text.split('.')[0]), image_width - 1), 0)
        bbox_ymax = max(min(int(bbox.find('ymax').text.split('.')[0]), image_height - 1), 0)
        
        if normalize:
            bbox = np.asarray([bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax], dtype = np.float32)
            bbox /= [image_width, image_height, image_width, image_height]
            bbox *= [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT]
            bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = bbox

        if (bbox_xmax - bbox_xmin) == 0 or (bbox_ymax - bbox_ymin) == 0:
            continue
        
        bboxes.append([bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax])
        classes.append(CLASS_DIC[label])

    return image_path, np.asarray(bboxes, dtype = np.float32), np.asarray(classes, dtype = np.int32)

def compute_bboxes_IoU(bboxes_1, bboxes_2):
    area_1 = (bboxes_1[:, 2] - bboxes_1[:, 0] + 1) * (bboxes_1[:, 3] - bboxes_1[:, 1] + 1)
    area_2 = (bboxes_2[:, 2] - bboxes_2[:, 0] + 1) * (bboxes_2[:, 3] - bboxes_2[:, 1] + 1)

    iw = np.minimum(bboxes_1[:, 2][:, np.newaxis], bboxes_2[:, 2]) - np.maximum(bboxes_1[:, 0][:, np.newaxis], bboxes_2[:, 0]) + 1
    ih = np.minimum(bboxes_1[:, 3][:, np.newaxis], bboxes_2[:, 3]) - np.maximum(bboxes_1[:, 1][:, np.newaxis], bboxes_2[:, 1]) + 1

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)
    
    intersection = iw * ih
    union = (area_1[:, np.newaxis] + area_2) - iw * ih

    return intersection / np.maximum(union, 1e-10)

def IoU(box1, box2):
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    
    area_xmin = np.maximum(box1[0], box2[0])
    area_ymin = np.maximum(box1[1], box2[1])
    area_xmax = np.minimum(box1[2], box2[2])
    area_ymax = np.minimum(box1[3], box2[3])

    area_w = np.maximum(0, area_xmax - area_xmin + 1)
    area_h = np.maximum(0, area_ymax - area_ymin + 1)

    intersection = area_w * area_h
    union = np.maximum(box1_area + box2_area - intersection, 1)

    return intersection / union

def py_nms(dets, thresh, mode = "Union"):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = np.arange(len(dets))

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if mode == "Union":
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == "Minimum":
            ovr = inter / np.minimum(areas[i], areas[order[1:]])

        #keep
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def class_nms(bboxes, classes, threshold = 0.5, mode = 'Union'):
    data_dic = {}
    nms_bboxes = []
    nms_classes = []

    for bbox, class_index in zip(bboxes, classes):
        try:
            data_dic[class_index].append(bbox)
        except KeyError:
            data_dic[class_index] = []
            data_dic[class_index].append(bbox)

    for key in data_dic.keys():
        data_dic[key] = np.asarray(data_dic[key], dtype = np.float32)
        keep_indexs = py_nms(data_dic[key], threshold)

        for bbox in data_dic[key][keep_indexs]:
            nms_bboxes.append(bbox)
            nms_classes.append(key)
    
    return nms_bboxes, nms_classes

def Precision_Recall(gt_boxes, gt_classes, pred_boxes, pred_classes, threshold_iou = 0.5):
    recall = 0.0
    precision = 0.0

    if len(gt_boxes) == 0:
        if len(pred_boxes) == 0:
            return 1.0, 1.0
        else:
            return 0.0, 0.0

    if len(pred_boxes) != 0:
        gt_boxes_cnt = len(gt_boxes)
        pred_boxes_cnt = len(pred_boxes)

        recall_vector = np.zeros(gt_boxes_cnt)
        precision_vector = np.zeros(pred_boxes_cnt)

        for gt_index in range(gt_boxes_cnt):
            for pred_index in range(pred_boxes_cnt):
                if IoU(pred_boxes[pred_index], gt_boxes[gt_index]) >= threshold_iou:
                    recall_vector[gt_index] = True
                    if gt_classes[gt_index] == pred_classes[pred_index]:
                        precision_vector[pred_index] = True

        recall = np.sum(recall_vector) / gt_boxes_cnt
        precision = np.sum(precision_vector) / pred_boxes_cnt

    return precision, recall
