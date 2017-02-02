import numpy as np
import tensorflow as tf

import tensorflow_vgg.vgg16 as vgg16
import tensorflow_vgg.utils as utils

from utils import gram_matrix

# import gram_matrix.gram_matrix as gram_matrix


def frobenious_norm(tensor):
    return tf.sqrt(tf.reduce_sum(tf.square(tensor)))

def style_loss_vgg16(vgg1, vgg2, ws):
    
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
    
    # style_layers_1 = tf.Variable([
            # vgg1.conv1_2,
            # vgg1.conv2_2,
            # vgg1.conv3_3 ])
            # # vgg1.conv4_3]) 

    # style_1 = tf.map_fn(gram_matrix, style_layers_1)
    
    # style_layers_2 = tf.Variable([
            # vgg2.conv1_2,
            # vgg2.conv2_2,
            # vgg2.conv3_3 ])
            # # vgg2.conv4_3])
    # style_2 = tf.map_fn(gram_matrix, style_layers_2)

    # styles_diff = tf.subtract(style_1, style_2)
    # per_layer_errors = tf.map_fn(frobenious_norm, styles_diff) 
    # return tf.reduce_sum(tf.multiply(per_layer_errors, ws))


if __name__ == "__main__":
    img1 = utils.load_image("./tensorflow_vgg/test_data/starry_night_crop.jpg")
    img1 = img1.reshape((1, 224, 224, 3))

    img2 = utils.load_image("./tensorflow_vgg/test_data/chicago_starry_night.jpg")
    img2 = img2.reshape((1, 224, 224, 3))

    # img3 = utils.load_image("./tensorflow_vgg/test_data/tiger.jpeg")
    # img3 = img3.reshape((1, 224, 224, 3))

    i1 = tf.placeholder("float", [1, 224, 224, 3])
    i2 = tf.placeholder("float", [1, 224, 224, 3])
    # i3 = tf.placeholder("float", [1, 224, 224, 3])

    feed_dict = {i1: img1, i2: img1}
    # feed_dict = {i1: img1, i2: img2, i3: img3}

    vgg1 = vgg16.Vgg16()
    vgg2 = vgg16.Vgg16()
    # vgg3 = vgg16.Vgg16()

    with tf.name_scope("content_vgg"):
        vgg1.build(i1)
        vgg2.build(i2)
        # vgg3.build(i3)

    style_weights = tf.Variable([1.0, 1.0, 1.0, 1.0])
    error = style_loss_vgg16(vgg1, vgg2, style_weights)

    initialize = tf.global_variables_initializer()


    with tf.Session(
            config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
        sess.run(initialize)
        
        e = sess.run(error, feed_dict=feed_dict)

