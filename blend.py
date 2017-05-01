import cv2
import numpy as np
from laplacian_blending import run_blend
import os.path as osp


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

def viz_pyramid(pyramid):
    """Create a single image by vertically stacking the levels of a pyramid."""
    shape = np.atleast_3d(pyramid[0]).shape[:-1]  # need num rows & cols only
    img_stack = [cv2.resize(layer, shape[::-1],
                            interpolation=3) for layer in pyramid]
    return np.vstack(img_stack).astype(np.uint8)
    
def laplacian_blend(white_image, black_image, mask):
    b_img = np.atleast_3d(black_image).astype(np.float) / 255.
    w_img = np.atleast_3d(white_image).astype(np.float) / 255.
    m_img = np.atleast_3d(mask).astype(np.float) / 255.
    num_channels = b_img.shape[-1]

    imgs = []
    for channel in range(num_channels):
        imgs.append(run_blend(b_img[:, :, channel],
                              w_img[:, :, channel],
                              m_img[:, :, channel]))

    names = ['gauss_pyr_black', 'gauss_pyr_white', 'gauss_pyr_mask',
             'lapl_pyr_black', 'lapl_pyr_white', 'outpyr', 'outimg']

    for name, img_stack in zip(names, zip(*imgs)):
        imgs = map(np.dstack, zip(*img_stack))
        stack = [cv2.normalize(img, img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX) for img in imgs]
        cv2.imwrite(name + '.png', viz_pyramid(stack))
    

for i in range(5):
    # Read images
    if i == 0:
        dst = cv2.imread("stylized.jpg")
    else:
        dst = cv2.imread("output.jpg")
    src = cv2.imread(osp.join("foreground", "seg_image_{0}.jpg".format(i+1))
    mask = cv2.imread("mask_{0}.jpg".format(i+1))
    # mask = cv2.imread("seg_patch.jpg")

    # print(mask.shape)

    # print("Foreground", src.shape)
    # print("Background", dst.shape)
    # print(mask)

    output = poisson_blend(src, dst, mask)
    # output = laplacian_blend(src, dst, mask)

    # Save result
    cv2.imwrite("output.jpg", output);
