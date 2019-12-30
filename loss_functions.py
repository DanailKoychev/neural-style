import tensorflow as tf


def frobenious_norm(tensor):
    return tf.sqrt(tf.reduce_sum(tf.square(tensor)))


def gram_matrix(features):
    # features should be a layer activation for single input,
    # so the shape should be [1, height, width, channels]
    shape = features.get_shape().as_list()
    features = tf.reshape(features, [shape[1] * shape[2], shape[3]])
    unnormalized_gram_m = tf.matmul(features, features, transpose_a=True)
    return tf.div(unnormalized_gram_m, shape[1] * shape[2] * shape[3])


def style_transfer_loss(
        vgg_var, vgg_content, vgg_style, weights,
        style_ws=tf.constant([1.0, 1.0, 1.0, 1.0])):
    content = content_loss(vgg_var, vgg_content)
    style = style_loss(vgg_var, vgg_style, style_ws)
    return tf.add(tf.multiply(content, weights[0]), tf.multiply(style, weights[1]))


def content_loss(vgg1, vgg2):
    content_layer_1 = vgg1.conv4_2
    content_layer_2 = vgg2.conv4_2
    dist = frobenious_norm(tf.subtract(content_layer_1, content_layer_2))
    shape = content_layer_1.get_shape().as_list()
    return tf.div(dist, shape[1] * shape[2] * shape[3]) 


def style_loss(vgg1, vgg2, ws):
    c1_2_i1 = vgg1.conv1_2
    c2_2_i1 = vgg1.conv2_2
    c3_3_i1 = vgg1.conv3_3
    c4_3_i1 = vgg1.conv4_3

    g1_1 = gram_matrix(c1_2_i1)
    g2_1 = gram_matrix(c2_2_i1)
    g3_1 = gram_matrix(c3_3_i1)
    g4_1 = gram_matrix(c4_3_i1)

    c1_2_i2 = vgg2.conv1_2
    c2_2_i2 = vgg2.conv2_2
    c3_3_i2 = vgg2.conv3_3
    c4_3_i2 = vgg2.conv4_3

    g1_2 = gram_matrix(c1_2_i2)
    g2_2 = gram_matrix(c2_2_i2)
    g3_2 = gram_matrix(c3_3_i2)
    g4_2 = gram_matrix(c4_3_i2)

    l1_diff = frobenious_norm(tf.subtract(g1_1, g1_2))
    l2_diff = frobenious_norm(tf.subtract(g2_1, g2_2))
    l3_diff = frobenious_norm(tf.subtract(g3_1, g3_2))
    l4_diff = frobenious_norm(tf.subtract(g4_1, g4_2))

    diffs = tf.pack([l1_diff, l2_diff, l3_diff, l4_diff])

    return tf.reduce_sum(tf.multiply(diffs, ws))
    
