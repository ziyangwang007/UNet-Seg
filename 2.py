import cv2 as cv
import numpy as np
import random
from PIL import ImageChops
import matplotlib.pyplot as plt
label = cv.imread('data/isic2017/train/masks/ISIC_0000002_segmentation.png')
print(np.mean(label))
print(label.shape)
cv.imshow('11',label)
cv.waitKey(0)
cv.destroyAllWindows()
edge = cv.Canny(label, 0.1, 0.2)
kernel = np.ones((4, 4), np.uint8)
if True:
    edge = edge[6:-6, 6:-6]
    edge = np.pad(edge, ((6,6),(6,6)), mode='constant')
edge = (cv.dilate(edge, kernel, iterations=1)>50)*1.0

# if True:
#     rand_scale = 0.5 + random.randint(0, 16) / 10.0
    # image, label, edge = self.multi_scale_aug(image, label, edge,
    #                                     rand_scale=rand_scale)
# img = Image.open("/home/newj/图片/space.jpeg")

# plt.figure("Image") # 图像窗口名称
# plt.imshow(img)
# cv2.imshow('11',edge)
# cv2.waitKey(0)