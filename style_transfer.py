import argparse
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import tensorflow as tf

import tensorflow_vgg.vgg16 as vgg16
import tensorflow_vgg.utils as utils
import vgg16_nofc

from loss_functions import *
from utils import *


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('content_path', type=str,
                                help='content image')
    parser.add_argument('style_path', type=str,
                                help='style image')
    parser.add_argument('--save_path', type=str, default=False, required=False,
                                help='path for the output image')
    parser.add_argument('--resolution', type=int, default=512, required=False,
            help='image resolution (square), eg: for 512x512: --resolution 512')
    parser.add_argument('--iterations', type=int, default=200, required=False,
                                help='nuber of iterations for the optimizer')
    parser.add_argument('--content_weight', type=float, default=1000, required=False,
                                help='relative weight of the content')
    parser.add_argument('--style_weight', type=float, default=1, required=False,
                                help='relative weight of the style')
    parser.add_argument('--verbose', action='store_const', const=True, required=False,
                                help='prints intermediate cost function values')
    return parser


if __name__ == "__main__":
    #parse arguments
    args = parse_arguments().parse_args()
    path_cont = args.content_path
    path_style = args.style_path
    content_weight = float(args.content_weight)
    style_weight = float(args.style_weight)
    iterations = args.iterations
    img_height = img_width = args.resolution 
    verbose = args.verbose
    style_layer_weights = [1.0, 1.0, 1.0, 1.0] 

    #load iamges
    img_content = utils.load_image(path_cont, img_width, img_height)
    img_content = img_content.reshape((1, img_width, img_height, 3))
    img_style = utils.load_image(path_style, img_width, img_height)
    img_style = img_style.reshape((1, img_width, img_height, 3))

    img_var = tf.Variable(img_content.astype("float32"), name="img_var")
    placeholder_content = tf.placeholder("float", [1, img_width, img_height, 3])
    placeholder_style = tf.placeholder("float", [1, img_width, img_height, 3])


    vgg_content = vgg16_nofc.Vgg16()
    vgg_content.build(placeholder_content)

    vgg_style = vgg16_nofc.Vgg16()
    vgg_style.build(placeholder_style)
    
    vgg_var = vgg16_nofc.Vgg16()
    vgg_var.build(img_var)

    style_layer_weights = tf.constant(style_layer_weights)
    loss_type_weights = tf.constant([content_weight, style_weight])
    error = total_loss(vgg_var, vgg_content, vgg_style, loss_type_weights,
            style_layer_weights)

    train_op = tf.train.AdamOptimizer(1e-1/5).minimize(error)
    feed_dict = {placeholder_content: img_content, placeholder_style: img_style}
    initialize = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(initialize)

        intermediate = []
        final_img = []

        for i in range(iterations):
            sess.run(train_op, feed_dict=feed_dict)
            final_img = img_var.eval()
            e = sess.run(error, feed_dict=feed_dict)

            if i % 10 == 0:
                intermediate.append(final_img)
                if verbose:
                    print('iter:', i, ' error:', e)

    fin = trim_colors(intermediate[-1])
    skimage.io.imshow(fin)
    plt.show()

    if args.save_path is not None:
        save(intermediate[-1], args.save_path)

