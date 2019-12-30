import os
import numpy as np
import skimage.transform
import skimage.io
from PIL import Image


def trim_colors(image):
    img = np.copy(image)
    shape = img.shape
    img.shape = (shape[1]*shape[2]*shape[3])
    np.minimum(img, 1, img)
    np.maximum(img, 0, img)
    img.shape = (shape[2], shape[1], shape[3])
    return img


def save_image(image, path):
    img = trim_colors(image)
    Image.fromarray(img).convert('RGBA').save(path)


def load_image(path):
    """ Loads an image cropped and rescaled to (224, 224)"""
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img

