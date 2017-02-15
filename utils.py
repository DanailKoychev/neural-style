import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import skimage.io

def trim_colors(image):
    img = np.copy(image)
    shape = img.shape
    img.shape = (shape[1]*shape[2]*shape[3])
    np.minimum(img, 1, img)
    np.maximum(img, 0, img)
    img.shape = (shape[2], shape[1], shape[3])
    return img

def show(image):
    img = trim_colors(image)
    skimage.io.imshow(img)
    plt.show()

def save(image, path):
    img = trim_colors(image)
    skimage.io.imshow(img)
    plt.savefig(path)

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
