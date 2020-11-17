# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:snms.py
# software: PyCharm

import tensorflow as tf
from util import cal_iou
import time


# TODO: apply soft nms in my projects
def soft_nms(boxes,
             scores,
             max_output_size,
             nms_threshold,
             discard_threshold,
             penalty_method='linear',
             sigma=0.5):
    """ implement soft nms by tensorflow2.2
        but i think soft nms will bring more hyper-parameter in our model

    Args:
        boxes:              (n, 4)
        scores:             (n,)
        max_output_size:    a tensor
        nms_threshold:      a tensor
                            ###############################
                            author uses 0.3 in experiments
                            ###############################
        discard_threshold:  a tensor
                            ###############################
                            usually be 10e-4 or 10e-2
                            ###############################
        penalty_method:     linear or gaussian
        sigma:              a scalar
                            author set it to 0.5 in experiments

    Returns:
        remain_id

    """
    scores_sorted = tf.argsort(scores)  # (n,)
    boxes_sorted = tf.gather(boxes, scores_sorted)

    while len(boxes_sorted) > 0:
        ious, _ = cal_iou(boxes_sorted[-1], boxes_sorted[:-1])  # (n-1,)

        if penalty_method == 'linear':
            soft_ids = tf.where(ious > nms_threshold)  # (num, 1)
            soft_ids_slice = soft_ids[..., 0]  # (num,)
            soft_ious = tf.gather(ious, soft_ids_slice)  # (num,)
            soft_score_ids = tf.gather(scores_sorted, soft_ids_slice)  # (num,)

            # extract scores
            soft_score = tf.gather(scores, soft_score_ids)  # (num,)
            soft_score = soft_score * (1 - soft_ious)
            # change scores elements
            scores = tf.tensor_scatter_nd_update(scores, tf.expand_dims(soft_score_ids, axis=1), soft_score)
        else:
            # in paper:
            # when overlap of a box bi with M becomes close to one,
            # bi should be significantly penalized. So, author proposes to use gaussian penalty function
            soft_score_ids = scores_sorted[:-1]
            # extract scores
            soft_score = tf.gather(scores, soft_score_ids)  # (num,)
            soft_score = soft_score * tf.exp(-tf.square(ious) / sigma)
            scores = tf.tensor_scatter_nd_update(scores, tf.expand_dims(soft_score_ids, axis=1), soft_score)

        # discard boxes that have score < discard threshold
        discard_id = tf.where(scores < discard_threshold)[..., 0]
        discard_id = tf.cast(discard_id, tf.int32)

        # re-sort scores and boxes
        scores_sorted = scores_sorted[:-1]

        insert_id = tf.sets.intersection(tf.expand_dims(scores_sorted, axis=0),
                                         tf.expand_dims(discard_id, axis=0)).values
        scores_sorted = tf.sets.difference(tf.expand_dims(scores_sorted, axis=0),
                                           tf.expand_dims(insert_id, axis=0)).values

        # boxes_sorted = tf.gather(boxes, scores_sorted)
        new_scores = tf.gather(scores, scores_sorted)
        resort_id = tf.argsort(new_scores)
        scores_sorted = tf.gather(scores_sorted, resort_id)
        boxes_sorted = tf.gather(boxes, resort_id)

    # choose boxes that have score > discard threshold
    # limit boxes with max output size
    scores_sort_id = tf.argsort(scores)
    scores = tf.gather(scores, scores_sort_id)
    remain_id = tf.where(scores > discard_threshold)[..., 0]
    remain_id = tf.gather(scores_sort_id, remain_id)[::-1]
    if tf.shape(remain_id) > max_output_size:
        remain_id = remain_id[0:max_output_size]

    return remain_id


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

    start = time.time()
    picked_ids = soft_nms(boxes_list,
                          scores_list,
                          10,
                          0.3,
                          10e-2,
                          penalty_method='linear')
    end = time.time()
    print(picked_ids)
    print('spend time:', end - start)
