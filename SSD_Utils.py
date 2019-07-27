import numpy as np
import tensorflow as tf

from Utils import *
from Define import *

def Generate_Anchors(layer_shapes, layer_aspect_ratios, min_max_scales):
    min_scale, max_scale = min_max_scales
    anchor_scales = np.linspace(min_scale, max_scale, num = len(layer_shapes))

    default_anchor_bboxes = []
    for index, anchor_scale, layer_shape, aspect_ratios in zip(range(len(SSD_LAYER_SHAPES)), anchor_scales, layer_shapes, layer_aspect_ratios):
        height, width = layer_shape

        for y in range(height):
            for x in range(width):
                for aspect_ratio in aspect_ratios:
                    # center x, y
                    anchor_cx = (x + 0.5) / width
                    anchor_cy = (y + 0.5) / height

                    # width, height
                    if aspect_ratio == -1:
                        anchor_width = np.sqrt(anchor_scale * anchor_scales[index + 1])
                        anchor_height = np.sqrt(anchor_scale * anchor_scales[index + 1])
                    else:
                        anchor_width = anchor_scale * np.sqrt(aspect_ratio)
                        anchor_height = anchor_scale / np.sqrt(aspect_ratio)
                        
                    default_anchor_bboxes.append([anchor_cx, anchor_cy, anchor_width, anchor_height])

    return np.asarray(default_anchor_bboxes, dtype = np.float32)

class SSD_EDCoder:
    def __init__(self, layer_shapes, aspect_ratios, min_max_scales, positive_iou_threshold):
        self.length = 0
        self.positive_iou_threshold = positive_iou_threshold
        self.default_anchor_bboxes = Generate_Anchors(layer_shapes, aspect_ratios, min_max_scales)

        for layer_shape, aspect_ratio in zip(layer_shapes, SSD_ASPECT_RATIOS):
            height, width = layer_shape
            self.length += (height * width * len(aspect_ratio))

        self.default_gt_classes = np.zeros((BATCH_SIZE, self.length, CLASSES))
        self.default_gt_classes[..., 0] = 1.

        print('[i] SSD_EDCoder')
        print('[i] length : {}, {}'.format(self.length, len(self.default_anchor_bboxes)))
        print('[i] positive_iou_threshold : {}'.format(self.positive_iou_threshold))
        print()

    def Encode(self, batch_data_list):
        gt_classes = self.default_gt_classes.copy()
        gt_offset_bboxes = np.zeros((BATCH_SIZE, self.length, 4))
        gt_positives = np.zeros((BATCH_SIZE, self.length))
        
        for batch_index, data_list in enumerate(batch_data_list):
            for data in data_list:
                gt_bbox, class_index = data
                bbox = xyxy_to_ccwh(gt_bbox)
                
                object_positive = False

                best_iou = -1
                best_anchor_index = 0
                best_gt_class = []
                best_gt_offset_bbox = []

                for anchor_index, anchor_bbox in enumerate(self.default_anchor_bboxes):
                    iou = IOU(bbox, anchor_bbox, 'center')
                    gt_class = one_hot(class_index, CLASSES)
                    gt_offset_bbox = get_offset_bbox(bbox, anchor_bbox)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_anchor_index = anchor_index

                        best_gt_class = gt_class
                        best_gt_offset_bbox = gt_offset_bbox
                    
                    if iou >= self.positive_iou_threshold:
                        object_positive = True
                        
                        gt_classes[batch_index][anchor_index] = gt_class
                        gt_offset_bboxes[batch_index][anchor_index] = gt_offset_bbox
                        gt_positives[batch_index][anchor_index] = 1

                if not object_positive:
                    gt_classes[batch_index][best_anchor_index] = best_gt_class
                    gt_offset_bboxes[batch_index][best_anchor_index] = best_gt_offset_bbox
                    gt_positives[batch_index][best_anchor_index] = 1

        return gt_classes, gt_offset_bboxes, gt_positives

    def Decode(self, pred_classes, pred_offset_bboxes, size = (IMAGE_WIDTH, IMAGE_HEIGHT), threshold = 0.5):
        bboxes = []
        classes = []

        image_width, image_height = size
        for anchor_index, anchor_bbox in enumerate(self.default_anchor_bboxes):
            pred_class_index = np.argmax(pred_classes[anchor_index])
            pred_class_prob = softmax(pred_classes[anchor_index])
            #pred_class_prob = pred_classes[anchor_index]

            if pred_class_index != 0 and pred_class_prob[pred_class_index] >= threshold:
                bbox = get_decode_bbox(pred_offset_bboxes[anchor_index], anchor_bbox)
                xmin, ymin, xmax, ymax = ccwh_to_xyxy(bbox) * [image_width, image_height, image_width, image_height]

                xmin = max(min(xmin, image_width - 1), 0)
                ymin = max(min(ymin, image_height - 1), 0)
                xmax = max(min(xmax, image_width - 1), 0)
                ymax = max(min(ymax, image_height - 1), 0)

                bboxes.append(np.append([xmin, ymin, xmax, ymax], pred_class_prob[pred_class_index]))
                classes.append(pred_class_index)

        return bboxes, classes

if __name__ == '__main__':
    import cv2

    ssd_edcoder = SSD_EDCoder(SSD_LAYER_SHAPES, SSD_ASPECT_RATIOS, [MIN_SCALE, MAX_SCALE], POSITIVE_IOU_THRESHOLD)

    batch_data_list = []
    bboxes = [[0.1, 0.1, 0.2, 0.2], [0.0, 0.0, 0.8, 0.8]]
    classes = [1, 2]

    data = []
    for bbox, class_index in zip(bboxes, classes):
        data.append([bbox, class_index])

    batch_data_list.append(data)

    gt_classes, gt_offset_bboxes, gt_positives = ssd_edcoder.Encode(batch_data_list)
    print(np.sum(gt_positives[0]))
    print(np.sum(1 - gt_positives[0]))

    test_image = np.zeros((360, 640, 3), dtype = np.uint8)
    pred_bboxes, pred_classes = ssd_edcoder.Decode(gt_classes[0], gt_offset_bboxes[0], size = (640, 360))

    print(len(pred_bboxes))
    for bbox, class_index in zip(pred_bboxes, pred_classes):
        xmin, ymin, xmax, ymax, conf = bbox
        xmin, ymin, xmax, ymax = [int(v) for v in [xmin, ymin, xmax, ymax]]
        cv2.rectangle(test_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)

    cv2.imshow('show', test_image)
    cv2.waitKey(0)
