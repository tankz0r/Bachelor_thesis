# -*- coding: utf-8 -*-
#!/bin/bash
from __future__ import division  # Python 2 users only

from keras import backend as K
import tensorflow as tf
import numpy as np
import keras.metrics as kmetrics

# The following metrics taken from Keras v1.2: precision, recall, fbeta_score, fmeasure
"""
All contributions by François Chollet:
Copyright (c) 2015, François Chollet.
All rights reserved.

All contributions by Google:
Copyright (c) 2015, Google, Inc.
All rights reserved.

All other contributions:
Copyright (c) 2015, the respective contributors.
All rights reserved.

Each contributor holds copyright over their respective contributions.
"""

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    """Computes the F score.

    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.

    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.

    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score
    
#---------------------------


def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.

    Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=1)


def append_metric(model, layer_num, metric_name, metric_tensor):
    """ Copied from Keras Model.compile so we can add a metric later """
    if len(model.output_names) > 1:
        metric_name = model.output_layers[layer_num].name + '_' + metric_name
    model.metrics_names.append(metric_name)
    model.metrics_tensors.append(metric_tensor)


def _brier(num_classes, preds, pick_classes):
    """
    Calculate the Brier skill score, from http://www.pnas.org/content/104/14/5959.full
    `preds` is the list of predicted probabilities, `pick_class` is the index of the class to use
    The reason we don't just take the maximum from `preds` is we might be using the `actual` value which
    might not be the predicted class
    """
    
    inv_num_cl = 1.0/num_classes
    
    denom = (1-inv_num_cl)**2 + (num_classes-1)*(inv_num_cl)**2
    numerator = 1 + K.sum(K.square(preds), axis=1)
    
    inds = tf.stack([tf.to_int64(tf.range(tf.shape(preds)[0])), pick_classes])
    t_inds = K.transpose(inds)
    sub_pick = 2*tf.gather_nd(preds, t_inds)    
    brier_pick = 1 - (numerator - sub_pick)/denom  
    
    return brier_pick


def brier_skill(y_true, y_pred, use_true):
    """
    Calculate Brier score, relative to either true class or predicted class
    if use_true = True, it's relative to the y_true class
    if use_true = False, it's relative to the y_pred class (how confident are we in the prediction, no knowledge of true class)
    
    """
    do_eval = False
    
    # We use this function later on for static values
    if isinstance(y_pred, np.ndarray):
        do_eval = True
        y_pred = K.variable(y_pred)
    
    num_classes = K.get_variable_shape(y_pred)[1]
    
    if use_true:
        y_pick = y_true
    else:
        y_pick = y_pred
        
    pick_classes = K.argmax(y_pick, axis=1)
    brier_out = _brier(num_classes, y_pred, pick_classes)
    
    if do_eval:
        brier_out = K.get_value(brier_out)
    
    return brier_out


def brier_pred(y_true, y_pred):
    return brier_skill(y_true, y_pred, False)


def brier_true(y_true, y_pred):
    return brier_skill(y_true, y_pred, True)


def make_binary_metric(metric_name, metric_func, num_classes, y_true, preds_one_hot):
    """Create a binary metric using `metric_func`"""
    overall_met = [None for _ in range(num_classes)]
    with tf.name_scope(metric_name):
        for cc in range(num_classes):
            #Metrics should take 1D arrays which are 1 for positive, 0 for negative
            two_true, two_pred = y_true[:, cc], preds_one_hot[:, cc]
            cur_met = metric_func(two_true, two_pred)
            tf.summary.scalar('%d' % cc, cur_met)

            overall_met[cc] = cur_met
                
        tf.summary.histogram('overall', overall_met) 


def create_batch_pairwise_metrics(y_true, y_pred):
    """Create precision, recall, and fmeasure metrics. Log them directly using tensorflow"""
    num_classes = K.get_variable_shape(y_pred)[1]
    preds_cats = K.argmax(y_pred, axis=1)
    preds_one_hot = K.one_hot(preds_cats, num_classes)
    
    make_binary_metric('precision', precision, num_classes, y_true, preds_one_hot)
    make_binary_metric('recall', recall, num_classes, y_true, preds_one_hot)
    make_binary_metric('fmeasure', fmeasure, num_classes, y_true, preds_one_hot)


def multi_to_two_class(one_hot, pos_class_num):
    """
    inarr in N x num_class, we change to N x 2
    where `pos_class_num` is the positive class and column 1,
    negative is column 0
    """

    # K = np
    pos_col = one_hot[:, pos_class_num]
    out_arr = K.transpose(K.stack([1 - pos_col, pos_col]))
    # out_arr = pos_col
    return out_arr


def make_stats(prefix, metric):
    import tensorflow as tf

    use_metric = metric
    if isinstance(metric, list):
        use_metric = tf.stack(metric)

    out_dict = {'%s/mean' % prefix: K.mean(use_metric),
                '%s/std' % prefix: K.std(use_metric),
                '%s/max' % prefix: K.max(use_metric),
                '%s/min' % prefix: K.min(use_metric)}
    return out_dict


def batch_pairwise_metrics(y_true, y_pred):
    # assert K.get_variable_shape(y_true)[1] == K.get_variable_shape(y_pred)[1]
    num_classes = K.get_variable_shape(y_pred)[1]
    preds_cats = K.argmax(y_pred, axis=1)
    preds_one_hot = K.one_hot(preds_cats, num_classes)

    overall_precision = [None for _ in range(num_classes)]
    overall_recall = [None for _ in range(num_classes)]
    overall_fmeasure = [None for _ in range(num_classes)]

    out_dict = {}
    for cc in range(num_classes):
        # Metrics should take 1D arrays which are 1 for positive, 0 for negative
        two_true, two_pred = y_true[:, cc], preds_one_hot[:, cc]
        cur_dict = {
            'precision/%02d' % cc: precision(two_true, two_pred),
            'recall/%02d' % cc: recall(two_true, two_pred),
            'fmeasure/%02d' % cc: fmeasure(two_true, two_pred),
            'binary_accuracy/%02d' % cc: kmetrics.binary_accuracy(two_true, two_pred),
            'act_pos/%02d' % cc: K.sum(two_true),
            'pred_pos/%02d' % cc: K.sum(two_pred)
        }
        out_dict.update(cur_dict)

        overall_precision[cc] = cur_dict['precision/%02d' % cc]
        overall_recall[cc] = cur_dict['recall/%02d' % cc]
        overall_fmeasure[cc] = cur_dict['fmeasure/%02d' % cc]

    out_dict.update(make_stats('precision', overall_precision))
    out_dict.update(make_stats('recall', overall_recall))
    out_dict.update(make_stats('fmeasure', overall_fmeasure))

    return out_dict