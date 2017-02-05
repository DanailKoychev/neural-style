import numpy as np
import os
import sys
import tensorflow as tf

import tensorflow_vgg.vgg16 as vgg16
import tensorflow_vgg.utils as utils

import vgg16_nofc
# from vgg16_nofc import Vgg16 as Vgg16

from style_loss import style_loss_vgg16 as style_loss
from content_loss import content_loss as content_loss
from utils import frobenious_norm

import skimage.io
import matplotlib.pyplot as plt

# from images import *  # imags for testing

img_width = 512  
img_height = 512


def total_loss(vgg_var, vgg_content, vgg_style, weights, style_ws=tf.constant([1.0, 1.0, 1.0, 1.0])):
    content = content_loss(vgg_var, vgg_content)
    style = style_loss(vgg_var, vgg_style, style_ws)
    return tf.add(tf.multiply(content, weights[0]), tf.multiply(style, weights[1]))


def trim_colors(image):
    img = np.copy(image)
    img.shape = (img_width*img_height*3)
    np.minimum(img, 1, img)
    np.maximum(img, 0, img)
    img.shape = (img_height, img_width, 3)
    return img

def show(image):
    img = trim_colors(image)
    skimage.io.imshow(img)
    plt.show()

def save(image, name):
    img = trim_colors(image)
    skimage.io.imshow(img)
    plt.savefig(name)


def save_a(images, directory, name):
    os.mkdir(directory)
    for i, image in enumerate(images):
       img = trim_colors(image)
       skimage.io.imshow(img)
       plt.savefig(directory + "/" + name + str(i))

def save_all(images, name):
    for i, image in enumerate(images):
        img = trim_colors(image)
        skimage.io.imshow(img)
        plt.savefig(name + "_" + str(i))


if __name__ == "__main__":

    img_sketch_2 = utils.load_image("./tensorflow_vgg/test_data/sketch2.jpg", img_width, img_height)
    img_sketch_2 = img_sketch_2.reshape((1, img_width, img_height, 3))

    chicago_starry = utils.load_image("./presentation/chrysanthemum2.jpg", img_width, img_height)
    chicago_starry = chicago_starry.reshape((1, img_width, img_height, 3))



    path_cont = sys.argv[1]
    path_style = sys.argv[2]

    img_content = utils.load_image(path_cont, img_width, img_height)
    img_content = img_content.reshape((1, img_width, img_height, 3))

    img_style = utils.load_image(path_style, img_width, img_height)
    img_style = img_style.reshape((1, img_width, img_height, 3))


    # img_content = img_natalie
    # img_style = img_sketch_2


    img_gen = tf.Variable(img_content.astype("float32"), name="img_gen")
    # img_gen = tf.Variable(chicago_starry.astype("float32"), name="img_gen")
    # img_gen = tf.Variable(img_sketch_2.astype("float32"), name="img_gen")
    # img_gen = tf.Variable(tf.random_uniform((1, img_height, img_width, 3), minval=0, maxval=1, dtype=tf.float32), name="img_gen")


    placehodler_content = tf.placeholder("float", [1, img_width, img_height, 3])
    placeholder_style = tf.placeholder("float", [1, img_width, img_height, 3])

    vgg_content = vgg16_nofc.Vgg16()
    vgg_style = vgg16_nofc.Vgg16()
    vgg_var = vgg16_nofc.Vgg16()


    with tf.name_scope("content_vgg"):
        vgg_content.build(placehodler_content)
        vgg_style.build(placeholder_style)
        vgg_var.build(img_gen)


    style_weights = tf.constant([1.0, 1.0, 2.0, 3.0])
    loss_type_weights = tf.constant([4e+5, 1.0])
    error = total_loss(vgg_var, vgg_content, vgg_style, loss_type_weights)


    train_op = tf.train.AdamOptimizer(1e-1/5).minimize(error)
    feed_dict = {placehodler_content: img_content, placeholder_style: img_style} # the image we generate is a variable and is not fed trhough the dict
    initialize = tf.global_variables_initializer()


    with tf.Session(
            config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:

        sess.run(initialize)
       
        intermediate = []
        final_img = []
        e_prev = 1000000000
        for i in range(200000):
            sess.run(train_op, feed_dict=feed_dict)
            final_img = img_gen.eval()
            e = sess.run(error, feed_dict=feed_dict)

            if np.abs(e_prev - e) < 50:
                print("possible convergence")
                # break
            e_prev = e

            if i % 10 == 0 or (i < 10):
                intermediate.append(final_img)
                print(i, e)

                # print(content_loss(img_gen, img_content))
                # print(style_loss(img_gen, img_width)

    fin = trim_colors(intermediate[-1])
    skimage.io.imshow(fin)
    plt.show()






