import numpy as np
import scipy as scp
from scipy import ndimage, misc
import argparse


def subtract_mask(img, mask):
    mask = mask / 255
    mask = np.stack((mask,)*3, axis=2)
    print(mask.shape)
    img = np.multiply(img, mask)
    misc.imshow(img)


img = ndimage.imread("deepmask/data/testImage.jpg")
mask = ndimage.imread("deepmask/mask.jpg")

new_img = subtract_mask(img, mask)
