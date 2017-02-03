import numpy as np
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



img_width = 650
img_height = 650




def total_loss_classic(vgg_var, vgg_content, vgg_style, weights, style_ws=tf.constant([1.0, 1.0, 1.0, 1.0])):
    content = content_loss(vgg_var, vgg_content)
    style = style_loss(vgg_var, vgg_style, style_ws)

    return tf.add(tf.multiply(content, weights[0]), tf.multiply(style, weights[1]))
    # loss = tf.concat(content, style)
    # return tf.multiply(loss, weights)


def trim_colors(image):
    img = np.copy(image)
    img.shape = (img_width*img_height*3)
    img = [x if x <= 1 else 1 for x in img]
    img = [x if x >= 0 else 0 for x in img]
    img = np.asarray(img)
    # img.shape = (img_width, img_height, 3)
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

def save_all(images, name):
    for i, image in enumerate(images):
        img = trim_colors(image)
        skimage.io.imshow(img)
        plt.savefig(name + "_" + str(i))
    

if __name__ == "__main__":
    # img_starry_night = utils.load_image("./tensorflow_vgg/test_data/starry_night_crop.jpg", img_width, img_height)
    # img_starry_night = img_starry_night.reshape((1, img_width, img_height, 3))

    # img_chicago_stary = utils.load_image("./tensorflow_vgg/test_data/chicago_starry_night.jpg", img_width, img_height)
    # img_chicago_stary = img_chicago_stary.reshape((1, img_width, img_height, 3))

    # img_tiger = utils.load_image("./tensorflow_vgg/test_data/tiger.jpeg", img_width, img_height)
    # img_tiger = img_tiger.reshape((1, img_width, img_height, 3))

    # img_chicago = utils.load_image("./tensorflow_vgg/test_data/chicago_original.jpg", img_width, img_height)
    # img_chicago = img_chicago.reshape((1, img_width, img_height, 3))

    # img_candy = utils.load_image("./tensorflow_vgg/test_data/candy.jpg", img_width, img_height)
    # img_candy = img_candy.reshape((1, img_width, img_height, 3))

    # img_scream = utils.load_image("./tensorflow_vgg/test_data/the-scream.jpg", img_width, img_height)
    # img_scream = img_scream.reshape((1, img_width, img_height, 3))

    # img_fmi = utils.load_image("./tensorflow_vgg/test_data/fmi.jpg", img_width, img_height)
    # img_fmi = img_fmi.reshape((1, img_width, img_height, 3))

    # img_oil_crop = utils.load_image("./tensorflow_vgg/test_data/fmi.jpg", img_width, img_height)
    # img_oil_crop = img_oil_crop.reshape((1, img_width, img_height, 3))

    # img_wave = utils.load_image("./tensorflow_vgg/test_data/wave_crop.jpg", img_width, img_height)
    # img_wave = img_wave.reshape((1, img_width, img_height, 3))
    
    # img_chrysanthemum = utils.load_image("./tensorflow_vgg/test_data/chrysanthemum.jpg", img_width, img_height)
    # img_chrysanthemum = img_chrysanthemum.reshape((1, img_width, img_height, 3))

    # img_chrysanthemum2 = utils.load_image("./tensorflow_vgg/test_data/chrysanthemum2.jpg", img_width, img_height)
    # img_chrysanthemum2 = img_chrysanthemum2.reshape((1, img_width, img_height, 3))

    # img_natalie = utils.load_image("./tensorflow_vgg/test_data/natalie_dormer.jpeg", img_width, img_height)
    # img_natalie = img_natalie.reshape((1, img_width, img_height, 3))

    # img_comic = utils.load_image("./tensorflow_vgg/test_data/comic_girl_face.jpg", img_width, img_height)
    # img_comic = img_comic.reshape((1, img_width, img_height, 3))

    # img_eiffel_tower = utils.load_image("./tensorflow_vgg/test_data/eiffel_tower.jpg", img_width, img_height)
    # img_eiffel_tower = img_eiffel_tower.reshape((1, img_width, img_height, 3))
    
    # img_chicago_224_512 = utils.load_image("chicago_224-512.jpg", img_width, img_height)
    # img_chicago_224_512 = img_chicago_224_512.reshape((1, img_width, img_height, 3))

    # img_fmi2 = utils.load_image("./tensorflow_vgg/test_data/fmi_2.jpg", img_width, img_height)
    # img_fmi2 = img_fmi2.reshape((1, img_width, img_height, 3))

    # img_jaguar1 = utils.load_image("./tensorflow_vgg/test_data/jaguar1.jpg", img_width, img_height)
    # img_jaguar1 = img_jaguar1.reshape((1, img_width, img_height, 3))

    # flame1 = utils.load_image("./tensorflow_vgg/test_data/flame1.jpg", img_width, img_height)
    # flame1 = flame1.reshape((1, img_width, img_height, 3))

    fire2 = utils.load_image("./tensorflow_vgg/test_data/fire2.jpg", img_width, img_height)
    fire2 = fire2.reshape((1, img_width, img_height, 3))

    # eagle1 = utils.load_image("./tensorflow_vgg/test_data/eagle1.jpg", img_width, img_height)
    # eagle1 = eagle1.reshape((1, img_width, img_height, 3))

    jaguar4 = utils.load_image("./tensorflow_vgg/test_data/jaguar4.jpg", img_width, img_height)
    jaguar4 = jaguar4.reshape((1, img_width, img_height, 3))

    # fire3 = utils.load_image("./tensorflow_vgg/test_data/fire3.jpg", img_width, img_height)
    # fire3 = fire3.reshape((1, img_width, img_height, 3))



    img_content = jaguar4
    img_style = fire2




    # gen_init = tf.random_normal([1,img_size,img_size,3])
    # img_gen = tf.Variable(gen_init, name="img_gen")
    # chicago_f64 = chicago.astype("float64")
    img_gen = tf.Variable(img_content.astype("float32"), name="img_gen")
    # img_gen = tf.Variable(img_chicago_224_512.astype("float32"), name="img_gen")
    
    # img_gen = tf.ones([1, img_size, img_size, 3])
    # img_gen = tf.Variable()
    # img_gen.set_shape([1, img_size, img_size, 3])


    # i1 = tf.placeholder("float", [1, img_width, img_height, 3])
    placehodler_content = tf.placeholder("float", [1, img_width, img_height, 3])
    placeholder_style = tf.placeholder("float", [1, img_width, img_height, 3])

    vgg1 = vgg16_nofc.Vgg16()
    vgg_content = vgg16_nofc.Vgg16()
    vgg_style = vgg16_nofc.Vgg16()

    vgg_var = vgg16_nofc.Vgg16()

    with tf.name_scope("content_vgg"):
        # vgg1.build(i1)
            # vgg1.build(img_gen)

        vgg_content.build(placehodler_content)
        vgg_style.build(placeholder_style)

        vgg_var.build(img_gen)

    style_weights = tf.constant([1.0, 1.0, 1.0, 1.0])
    loss_type_weights = tf.constant([500.0, 1.0])
    error = total_loss_classic(vgg_var, vgg_content, vgg_style, loss_type_weights)
    # error2 = total_loss_classic(vgg1, vgg_content, vgg_style, loss_type_weights)


    train_op = tf.train.AdamOptimizer(1e-1/5).minimize(error)
    # feed_dict = {i1: img_gen, placehodler_content: chicago, placeholder_style: img_starry_night}
    # feed_dict = {i1: chicago, placehodler_content: img_starry_night, placeholder_style: img_gen}
    # feed_dict = {i1: chicago_stary, placehodler_content: chicago, placeholder_style: img_starry_night} # the first img is var and is not "fed" trhough the dict
    feed_dict = {placehodler_content: img_content, placeholder_style: img_style} # the first img is var and is not "fed" trhough the dict
    initialize = tf.global_variables_initializer()


    with tf.Session(
            config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
        sess.run(initialize)
       
        intermediate = []
        final_img = []
        e_prev = 1000000000
        for i in range(200000):
        # for i in range(20):
            sess.run(train_op, feed_dict=feed_dict)
            final_img = img_gen.eval()
            e = sess.run(error, feed_dict=feed_dict)

            if np.abs(e_prev - e) < 50:
                print("CONVERGEEEEEENNNNCEEEEE!!!!!! (mabye?)")
                # break
            e_prev = e

            if i % 10 == 0 or (i < 10):
                intermediate.append(final_img)
                print(i, e)

    fin = np.copy(intermediate[-1])
    fin.shape = (img_size*img_size*3)
    fin = [x if x <= 1 else 1 for x in fin]
    fin = [x if x >= 0 else 0 for x in fin]
    fin = np.asarray(fin)
    fin.shape = (img_size, img_size, 3)

    # fin = np.copy(final_img)
    # fin.shape = (img_size * img_size * 3)
    # fin -= min(fin)
    # fin /= max(fin)
    # fin.shape = (img_size, img_size, 3)

    
    skimage.io.imshow(fin)
    plt.show()
    # skimage.io.imshow(fin)


        # e = sess.run(error, feed_dict=feed_dict)
        # # e = sess.run(vgg_var.conv1_1, feed_dict=feed_dict)
        # print(e)
        
        # vgg_var = vgg16.Vgg16() 
        # vgg_var.build(i1)
        # error = total_loss_classic(vgg_var, vgg_content, vgg_style, loss_type_weights)
        # e = sess.run(error, feed_dict=feed_dict)
        # print(e)    







        



