# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:diou_nms.py
# software: PyCharm

import tensorflow as tf
from util import cal_iou


def diou_nms(boxes,
             scores,
             max_output_size,
             diou_threshold):
    """

    Args:
        boxes:            (n, 4)
        scores:           (n,)
        max_output_size:  scalar
        diou_threshold:   scalar

    Returns:
        pick_id: a tensor

    """
    scores_sorted = tf.argsort(scores)
    boxes_sorted = tf.gather(boxes, scores_sorted)

    picked = []

    while len(boxes_sorted) > 0:

        if len(picked) >= max_output_size:
            break

        picked.append(scores_sorted[-1])

        _, dious = cal_iou(boxes_sorted[-1], boxes_sorted[:-1])

        remain_ids = tf.where(dious < diou_threshold)[..., 0]
        scores_sorted = tf.gather(scores_sorted, remain_ids)
        boxes_sorted = tf.gather(boxes_sorted, remain_ids)

    picked = tf.cast(picked, tf.int32)
    return picked
