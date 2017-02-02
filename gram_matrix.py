import tensorflow as tf


def gram_matrix(hwc):
    # hwc should be a layer activation for single input,
    # so the shape should be [1, height, width, channels]
    shape = hwc.get_shape().as_list()
    hwc = tf.reshape(hwc, [shape[1] * shape[2], shape[3]])
    unnormalized_gram_m = tf.matmul(hwc, hwc, transpose_a=True)
    return tf.div(unnormalized_gram_m, shape[1] * shape[2] * shape[3])

    # shape = c22.get_shape().as_list()
    # c22 = tf.reshape(c22, [shape[1] * shape[2], shape[3]])
    # gm = tf.matmul(c22, c22, transpose_a=True)
    # print(gm)
