# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:util.py
# software: PyCharm

import tensorflow as tf


def cal_iou(box1, box2):
    """calculate iou

    Args:
        box1: (4,) a tensor
        box2: (n, 4[y1, x1, y2, x2]) a tensor

    Returns:
        ious
        dious

    """
    box1_min = box1[:2]
    box1_max = box1[2:4]
    box2_min = box2[..., :2]
    box2_max = box2[..., 2:4]

    insert_min = tf.maximum(box1_min, box2_min)
    insert_max = tf.minimum(box1_max, box2_max)

    box1_hw = box1_max - box1_min
    box2_hw = box2_max - box2_min
    box1_area = box1_hw[0] * box1_hw[1]
    box2_area = box2_hw[..., 0] * box2_hw[..., 1]
    insert_hw = tf.maximum(insert_max - insert_min, 0)
    insert_area = insert_hw[..., 0] * insert_hw[..., 1]

    ious = insert_area / (box1_area + box2_area - insert_area)

    box1_center = (box1_min + box2_max) / 2.0
    box2_center = (box2_min + box2_max) / 2.0
    center_distance = tf.sqrt(tf.square(box1_center - box2_center))
    diagonal_min = tf.minimum(box1_min, box2_min)
    diagonal_max = tf.maximum(box1_max, box2_max)
    diag_distance = tf.sqrt(tf.square(diagonal_min - diagonal_max))
    dious = ious - center_distance / (diag_distance + tf.keras.backend.epsilon)

    return ious, dious
