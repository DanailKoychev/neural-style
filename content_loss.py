import skimage.io

import numpy as np
import tensorflow as tf

import tensorflow_vgg.vgg16 as vgg16
import tensorflow_vgg.utils as utils

from utils import frobenious_norm


def content_loss(vgg1, vgg2):
    content_layer_1 = vgg1.conv4_2
    content_layer_2 = vgg2.conv4_2
    dist = frobenious_norm(tf.subtract(content_layer_1, content_layer_2))
    shape = content_layer_1.get_shape().as_list()
    return tf.div(dist, shape[1] * shape[2] * shape[3]) 


if __name__ == "__main__":
    img1 = utils.load_image("./tensorflow_vgg/test_data/starry_night_crop.jpg")
    img1 = img1.reshape((1, 224, 224, 3))

    img2 = utils.load_image("./tensorflow_vgg/test_data/chicago_starry_night.jpg")
    img2 = img2.reshape((1, 224, 224, 3))

    img3 = utils.load_image("./tensorflow_vgg/test_data/tiger.jpeg")
    img3 = img3.reshape((1, 224, 224, 3))

    img4 = utils.load_image("./tensorflow_vgg/test_data/chicago_original.jpg")
    img4 = img4.reshape((1, 224, 224, 3))

    i1 = tf.placeholder("float", [1, 224, 224, 3])
    i2 = tf.placeholder("float", [1, 224, 224, 3])
    # i3 = tf.placeholder("float", [1, 224, 224, 3])

    feed_dict = {i1: img2, i2: img4}
    # feed_dict = {i1: img1, i2: img2, i3: img3}

    vgg1 = vgg16.Vgg16()
    vgg2 = vgg16.Vgg16()
    # vgg3 = vgg16.Vgg16()

    with tf.name_scope("content_vgg"):
        vgg1.build(i1)
        vgg2.build(i2)
        # vgg3.build(i3)

    # style_weights = tf.Variable([1.0, 1.0, 1.0, 1.0])
    error = content_loss(vgg1, vgg2)

    initialize = tf.global_variables_initializer()


    with tf.Session(
            config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
        sess.run(initialize)
        
        e = sess.run(error, feed_dict=feed_dict)
        
