import numpy as np
import scipy as scp
from scipy import ndimage, misc
import argparse
import sys
import os.path as osp


def subtract_mask(img, mask, foreground=True):
    if foreground:
        mask = mask > 128
    else:
        mask = mask < 128
    mask = mask.astype(np.uint8)
    mask = np.stack((mask,)*3, axis=2)
    img = np.multiply(img, mask)
    if not foreground:
        img = img + (1 - mask).astype(np.uint8)*255

    return img

img = ndimage.imread(sys.argv[1])
num_masks = int(sys.argv[2])


for i in range(num_masks):
    mask = ndimage.imread(osp.join("masks", "mask_{0}.jpg".format(i+1)))

    fore_img = subtract_mask(img, mask)
    back_img = subtract_mask(img, mask, foreground=False)

    misc.imsave(osp.join("foreground", "seg_image_{0}.jpg".format(i+1)), fore_img)
    misc.imsave(osp.join("background", "seg_image_{0}.jpg".format(i+1)), back_img)

