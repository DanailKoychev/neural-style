import numpy as np
import tensorflow as tf

import tensorflow_vgg.vgg16 as vgg16
import tensorflow_vgg.utils as utils

from style_loss import style_loss_vgg16 as style_loss
from content_loss import content_loss as content_loss
from utils import frobenious_norm


def total_loss_classic(vgg_var, vgg_content, vgg_style, weights, style_ws=tf.constant([1.0, 1.0, 1.0, 1.0])):
    content = content_loss(vgg_var, vgg_content)
    style = style_loss(vgg_var, vgg_style, style_ws)

    return tf.add(tf.multiply(content, weights[0]), tf.multiply(style, weights[1]))
    # loss = tf.concat(content, style)
    # return tf.multiply(loss, weights)


if __name__ == "__main__":
    img1 = utils.load_image("./tensorflow_vgg/test_data/starry_night_crop.jpg")
    img1 = img1.reshape((1, 224, 224, 3))

    img2 = utils.load_image("./tensorflow_vgg/test_data/chicago_starry_night.jpg")
    img2 = img2.reshape((1, 224, 224, 3))

    img3 = utils.load_image("./tensorflow_vgg/test_data/tiger.jpeg")
    img3 = img3.reshape((1, 224, 224, 3))

    img4 = utils.load_image("./tensorflow_vgg/test_data/chicago_original.jpg")
    img4 = img4.reshape((1, 224, 224, 3))

    img_candy = utils.load_image("./tensorflow_vgg/test_data/candy.jpg")
    img_candy = img_candy.reshape((1, 224, 224, 3))

    img_scream = utils.load_image("./tensorflow_vgg/test_data/the-scream.jpg")
    img_scream = img_scream.reshape((1, 224, 224, 3))

    img_fmi = utils.load_image("./tensorflow_vgg/test_data/fmi.jpg")
    img_fmi = img_fmi.reshape((1, 224, 224, 3))

    # gen_init = tf.random_normal([1,224,224,3])
    # img_gen = tf.Variable(gen_init, name="img_gen")
    # img4_f64 = img4.astype("float64")
    img_gen = tf.Variable(img4.astype("float32"), name="img_gen")

    # img_gen = tf.ones([1, 224, 224, 3])
    # img_gen = tf.Variable()
    # img_gen.set_shape([1, 224, 224, 3])


    i1 = tf.placeholder("float", [1, 224, 224, 3])
    i2 = tf.placeholder("float", [1, 224, 224, 3])
    i3 = tf.placeholder("float", [1, 224, 224, 3])

    vgg1 = vgg16.Vgg16()
    vgg2 = vgg16.Vgg16()
    vgg3 = vgg16.Vgg16()

    vggg = vgg16.Vgg16()

    with tf.name_scope("content_vgg"):
        # vgg1.build(i1)
            # vgg1.build(img_gen)

        vgg2.build(i2)
        vgg3.build(i3)

        vggg.build(img_gen)

    style_weights = tf.constant([1.0, 1.0, 1.0, 1.0])
    loss_type_weights = tf.constant([70000.0, 1.0])
    error = total_loss_classic(vggg, vgg2, vgg3, loss_type_weights)
    # error2 = total_loss_classic(vgg1, vgg2, vgg3, loss_type_weights)


    train_op = tf.train.AdamOptimizer(1e-3).minimize(error)
    # feed_dict = {i1: img_gen, i2: img4, i3: img1}
    # feed_dict = {i1: img4, i2: img1, i3: img_gen}
    # feed_dict = {i1: img2, i2: img4, i3: img1} # the first img is var and is not "fed" trhough the dict
    feed_dict = {i1: img2, i2: img_fmi, i3: img_scream} # the first img is var and is not "fed" trhough the dict
    initialize = tf.global_variables_initializer()


    with tf.Session(
            config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
        sess.run(initialize)
        
        final_img = []
        for i in range(2000):
        # for i in range(20):
            sess.run(train_op, feed_dict=feed_dict)
            final_img = img_gen.eval()
            if i % 10 == 0:
                print(i, sess.run(error, feed_dict=feed_dict))

    fin = np.copy(final_img)
    fin.shape = (224*224*3)
    fin = [x if x <= 1 else 1 for x in fin]
    fin = [x if x >= 0 else 0 for x in fin]
    fin = np.asarray(fin)
    fin.shape = (224, 224, 3)

    # fin = np.copy(final_img)
    # fin.shape = (224 * 224 * 3)
    # fin -= min(fin)
    # fin /= max(fin)
    # fin.shape = (224, 224, 3)

    import skimage.io
    import matplotlib.pyplot as plt
    
    skimage.io.imshow(fin)
    plt.show()
    skimage.io.imshow(fin)


        # e = sess.run(error, feed_dict=feed_dict)
        # # e = sess.run(vggg.conv1_1, feed_dict=feed_dict)
        # print(e)
        
        # vggg = vgg16.Vgg16() 
        # vggg.build(i1)
        # error = total_loss_classic(vggg, vgg2, vgg3, loss_type_weights)
        # e = sess.run(error, feed_dict=feed_dict)
        # print(e)    







        



