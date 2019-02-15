import tensorflow as tf
#  sigma为3，计算smooth_l1损失
#  bbox_pred,偏移信息
#  bbox_targets,位置信息
#  bbox_inside_weights,内部信息
#  bbox_outside_weights外部新
#  https://blog.csdn.net/qq_26898461/article/details/52442145
def _modified_smooth_l1(sigma, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights):
    """
        ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
        SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                      |x| - 0.5 / sigma^2,    otherwise
    """
    # 9
    sigma2 = sigma * sigma
    # tf.subtract(bbox_pred, bbox_targets)，从bbox_pred减去bbox_targets，再与bbox_inside_weights相乘
    inside_mul = tf.multiply(bbox_inside_weights, tf.subtract(bbox_pred, bbox_targets))

    # 判断abs（inside_mul）是否小于1/9,如果小于对应位置返回True，否则为False，再tf.cast转换为0和1
    # 不理解？？？？？
    smooth_l1_sign = tf.cast(tf.less(tf.abs(inside_mul), 1.0 / sigma2), tf.float32)
    smooth_l1_option1 = tf.multiply(tf.multiply(inside_mul, inside_mul), 0.5 * sigma2)
    smooth_l1_option2 = tf.subtract(tf.abs(inside_mul), 0.5 / sigma2)

    # 结果就是实现上面的SmoothL1(x)结果
    smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                              tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))

    # 实现：bbox_outside_weights*SmoothL1(x)
    outside_mul = tf.multiply(bbox_outside_weights, smooth_l1_result)

    return outside_mul