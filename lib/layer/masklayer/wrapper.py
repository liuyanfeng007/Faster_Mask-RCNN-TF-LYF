import tensorflow as tf
from . import roi
from . import mask
def roi_decoder(boxes, scores, rois, ih, iw, scope='ROIDecoder'):
    with tf.name_scope(scope) as sc:
        final_boxes, classes, scores = tf.py_func(roi.decode,[boxes, scores, rois, ih, iw],[tf.float32, tf.int32, tf.float32])
        final_boxes = tf.convert_to_tensor(final_boxes, name='boxes')
        classes = tf.convert_to_tensor(tf.cast(classes, tf.int32), name='classes')
        scores = tf.convert_to_tensor(scores, name='scores')
        final_boxes = tf.reshape(final_boxes, (-1, 4))

    return final_boxes, classes, scores


def mask_encoder(gt_masks, gt_boxes, rois, num_classes, mask_height, mask_width, scope='MaskEncoder'):
    with tf.name_scope(scope) as sc:
        labels, mask_targets, mask_inside_weights = \
            tf.py_func(mask.encode,[gt_masks, gt_boxes, rois, num_classes, mask_height, mask_width],[tf.float32, tf.int32, tf.float32])
        labels = tf.convert_to_tensor(tf.cast(labels, tf.int32), name='classes')
        mask_targets = tf.convert_to_tensor(mask_targets, name='mask_targets')
        mask_inside_weights = tf.convert_to_tensor(mask_inside_weights, name='mask_inside_weights')
        labels = tf.reshape(labels, (-1,))
        mask_targets = tf.reshape(mask_targets, (-1, mask_height, mask_width, num_classes))
        mask_inside_weights = tf.reshape(mask_inside_weights, (-1, mask_height, mask_width, num_classes))
    return labels, mask_targets, mask_inside_weights