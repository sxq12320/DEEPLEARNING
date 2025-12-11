import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time

##滤波以及平滑处理
img = cv.imread('0.picture\p1.png')
print(img.shape)
img = img[40:550,0:700,:]

blur = cv.blur(img,(3,3))    #均值滤波
box = cv.boxFilter(img,-1,(3,3),normalize=True)  #方框滤波和上面差不多，-1表示颜色通道一致,True是有无做归一化，越界直接255了
aussian = cv.GaussianBlur(img,(5,5),1)     #高斯滤波
median = cv.medianBlur(img,5)

img_blur = cv.blur(img,(3,3))
img_blur2= cv.blur(img_blur,(3,3))
img_blur2_aussian = cv.GaussianBlur(img_blur2,(5,5),1)
res = cv.medianBlur(img_blur2_aussian,5)
res = cv.GaussianBlur(res,(3,3),2)

for i in range(1,5,1):
    res = cv.medianBlur(res , 5)
    res = cv.GaussianBlur(res , (5,5), 1)
    res = cv.blur(res,(3,3))

plt.subplot(231),    plt.imshow(img,'gray'),      plt.title('orginal picture')
plt.subplot(232),    plt.imshow(blur,'gray'),     plt.title('blur')
plt.subplot(233),    plt.imshow(box,'gray'),      plt.title('boxfilter')
plt.subplot(234),    plt.imshow(aussian,'gray'),  plt.title('aussian')
plt.subplot(235),    plt.imshow(median,'gray'),   plt.title('median')
plt.subplot(236),    plt.imshow(res,'gray'),   plt.title('mix')

plt.show()