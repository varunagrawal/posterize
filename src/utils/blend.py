import cv2
import numpy as np
from laplacian_blending import run_blend
import os.path as osp
import sys


def poisson_blend(fore, back, mask):
    mask = (mask > 128).astype(np.uint8) * 255
    back_mask = (mask < 128).astype(np.uint8)
    # back = np.multiply(back, back_mask) + (()back[50, 50, 0])

    # cv2.imwrite("masked_back.jpg", back)

    indices = np.argwhere(mask > 128)
    maxx, maxy = 0, 0
    minx, miny = mask.shape[1], mask.shape[0]
    x, y = 0, 0
    for i in indices:
        y += i[0]
        x += i[1]
        maxx = max(maxx, i[1])
        minx = min(minx, i[1])
        maxy = max(maxy, i[0])
        miny = min(miny, i[0])
    
    x = (maxx + minx) // 2
    y = (maxy + miny) // 2

    # cv2.imwrite("mask_patch.jpg", mask[miny:maxy, minx:maxx])
    # cv2.imwrite("seg_patch.jpg", fore[miny:maxy, minx:maxx])
    center = (x, y)
    # print(center)
    # print((back.shape[1]//2, back.shape[0]//2))
    # print((mask.shape[1]//2, mask.shape[0]//2))
    # Clone seamlessly.
    output = cv2.seamlessClone(fore, back, mask, center, cv2.NORMAL_CLONE)
    # print("Poisson Blending done.")
    return output


for i in range(int(sys.argv[1])):
    print("{0}...".format(i))
    # Read images
    if i == 0:
        dst = cv2.imread("stylized.jpg")
    else:
        dst = cv2.imread("output.jpg")
    src = cv2.imread(osp.join("foreground", "seg_image_{0}.jpg".format(i+1)))
    try:
        mask = cv2.imread(osp.join("masks", "mask_{0}.jpg".format(i+1)))
        # mask = cv2.imread("seg_patch.jpg")
    except:
        continue

    # print(mask.shape)

    # print("Foreground", src.shape)
    # print("Background", dst.shape)
    # print(mask)

    output = poisson_blend(src, dst, mask)
    # output = laplacian_blend(src, dst, mask)

    # Save result
    cv2.imwrite("output.jpg", output);
