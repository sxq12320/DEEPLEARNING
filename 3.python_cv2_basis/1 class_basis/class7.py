import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time

#图像阈值的操作处理

img_cat = cv.imread('0.picture\cat.jpg')
Img_Cat_Gray = cv.cvtColor(img_cat,cv.COLOR_BGR2GRAY)

ret,dst1 = cv.threshold(Img_Cat_Gray,150,255,cv.THRESH_BINARY)        #二值法，大于阈值（128）取最大值（255第三个参数），小于阈值取0
ret,dst2 = cv.threshold(Img_Cat_Gray,150,255,cv.THRESH_BINARY_INV)    #和而执法完全相反
ret,dst3 = cv.threshold(Img_Cat_Gray,128,255,cv.THRESH_TRUNC)         #截断之，大于128的直接等于128，小于128的直接不进行变化
ret,dst4 = cv.threshold(Img_Cat_Gray,128,255,cv.THRESH_TOZERO)        #大于的部分直接不变，小于等于128的直接成为0
ret,dst5 = cv.threshold(Img_Cat_Gray,128,255,cv.THRESH_TOZERO_INV)    #和上面相反

img = [img_cat,Img_Cat_Gray,dst1,dst2,dst3,dst4,dst5]
title=["Orginal image","THRESH_BINARY","THRESH_BINARY_INV","THRESH_TRUNC","THRESH_TOZERO","THRESH_TOZERO_INV"]

# for i in range(0,6):
#     plt.imshow(img[i],'gray')
#     plt.title(title[i])
#     plt.xticks([])
#     plt.yticks([])
#     plt.show()
#     time.sleep(1)
#     plt.close()
    
plt.subplot(231),plt.imshow(Img_Cat_Gray,'gray'),plt.title("Orginal image"),plt.xticks([]),plt.yticks([])
plt.subplot(232),plt.imshow(dst1,'gray'),plt.title("THRESH_BINARY"),plt.xticks([]),plt.yticks([])
plt.subplot(233),plt.imshow(dst2,'gray'),plt.title("THRESH_BINARY_INV"),plt.xticks([]),plt.yticks([])
plt.subplot(234),plt.imshow(dst3,'gray'),plt.title("THRESH_TRUNC"),plt.xticks([]),plt.yticks([])
plt.subplot(235),plt.imshow(dst4,'gray'),plt.title("THRESH_TOZERO"),plt.xticks([]),plt.yticks([])
plt.subplot(236),plt.imshow(dst5,'gray'),plt.title("THRESH_TOZERO_INV"),plt.xticks([]),plt.yticks([])
plt.show()