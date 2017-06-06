import numpy as np
import cv2
import sys
import os.path as osp
from matplotlib import pyplot as plt


img = cv2.imread(sys.argv[1])

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for i, (x,y,w,h) in enumerate(faces):
    # print(x, y, w, h)
    mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    # print(mask.shape)
    # mask[y:y+h, x:x+w] = 1
    rect = (x, y, w, h)

    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    # print(mask2)
    mask = mask * 255
    cv2.imwrite(osp.join("masks", "mask_{0}.jpg".format(i+1)), mask)
    # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    # roi_gray = gray[y:y+h, x:x+w]
    # roi_color = img[y:y+h, x:x+w]

    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

with open(osp.join("masks", "num_masks"), 'w') as f:
    f.write(str(len(faces)))

cv2.imwrite('faces_img.jpg',img)
print("Face Detection Done!")
