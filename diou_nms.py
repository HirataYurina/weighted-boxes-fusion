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
    """ implement diou nms by tensorflow2.2

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


if __name__ == '__main__':

    # test diou nms
    boxes_list = [
        [0.00, 0.51, 0.81, 0.91],
        [0.10, 0.31, 0.71, 0.61],
        [0.01, 0.32, 0.83, 0.93],
        [0.02, 0.53, 0.11, 0.94],
        [0.03, 0.24, 0.12, 0.35],
    ]
    scores_list = [0.9, 0.8, 0.2, 0.4, 0.7]

    boxes_list = tf.cast(boxes_list, tf.float32)
    scores_list = tf.cast(scores_list, tf.float32)

    pick_ids = diou_nms(boxes_list,
                        scores_list,
                        max_output_size=10,
                        diou_threshold=0.3)

    print(pick_ids)