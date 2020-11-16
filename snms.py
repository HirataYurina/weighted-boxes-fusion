# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:snms.py
# software: PyCharm

import tensorflow as tf
from util import cal_iou


def soft_nms(boxes,
             scores,
             max_output_size,
             nms_threshold,
             discard_threshold):
    """ implement soft nms by tensorflow2.2

    Args:
        boxes:              (n, 4)
        scores:             (n,)
        max_output_size:    a tensor
        nms_threshold:      a tensor
        discard_threshold:  a tensor
                            usually be 10e-4 or 10e-2

    Returns:
        remain_id

    """
    scores_sorted = tf.argsort(scores)  # (n,)
    boxes_sorted = tf.gather(boxes, scores_sorted)

    while len(boxes_sorted) > 0:
        ious, _ = cal_iou(boxes_sorted[-1], boxes_sorted[:-1])  # (n-1,)
        soft_ids = tf.where(ious > nms_threshold)  # (num, 1)
        soft_ids_slice = soft_ids[..., 0]  # (num,)
        soft_ious = tf.gather(ious, soft_ids_slice)  # (num,)
        soft_score_ids = tf.gather(scores_sorted, soft_ids_slice)  # (num,)

        # extract scores
        soft_score = tf.gather(scores, soft_score_ids)  # (num,)
        soft_score = soft_score * (1 - soft_ious)
        # change scores elements
        scores = tf.tensor_scatter_nd_update(scores, tf.expand_dims(soft_score_ids, axis=1), soft_score)

        # discard boxes that have score < discard threshold
        discard_id = tf.where(scores < discard_threshold)[..., 0]

        # re-sort scores and boxes
        scores_sorted = scores_sorted[:-1]

        insert_id = tf.sets.intersection(scores_sorted, discard_id)
        scores_sorted = tf.sets.difference(scores_sorted, insert_id)

        boxes_sorted = tf.gather(boxes, scores_sorted)
        new_scores = tf.gather(scores, scores_sorted)
        resort_id = tf.argsort(new_scores)
        scores_sorted = tf.gather(scores_sorted, resort_id)
        boxes_sorted = tf.gather(boxes_sorted, resort_id)

    # choose boxes that have score > discard threshold
    # limit boxes with max output size
    scores_sort_id = tf.argsort(scores)
    scores = tf.gather(scores, scores_sort_id)
    remain_id = tf.where(scores > discard_threshold)[..., 0]
    remain_id = tf.gather(scores_sort_id, remain_id)[::-1]
    if tf.shape(remain_id) > max_output_size:
        remain_id = remain_id[0:max_output_size]

    return tf.cast(remain_id, dtype=tf.int32)
