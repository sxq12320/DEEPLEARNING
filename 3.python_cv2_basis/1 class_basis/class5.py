import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

top_size,bottom_size,left_size,right_size = (50,50,50,50)

img = cv.imread('0.picture\cat.jpg')

#边缘填充

first = cv.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,borderType=cv.BORDER_REFLECT)      #复制方法
Second = cv.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,borderType=cv.BORDER_REPLICATE)  #反射法
third = cv.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,borderType=cv.BORDER_REFLECT101)   #反射法
forth = cv.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,borderType=cv.BORDER_WRAP)         #外包装法
fifth = cv.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,borderType=cv.BORDER_CONSTANT,value=2)     #常量法

plt.subplot(231), plt.imshow(img,'gray'), plt.title('1')
plt.subplot(232), plt.imshow(first,'gray'), plt.title('2')
plt.subplot(233), plt.imshow(Second,'gray'), plt.title('3')
plt.subplot(234), plt.imshow(third,'gray'), plt.title('4')
plt.subplot(235), plt.imshow(forth,'gray'), plt.title('5')
plt.subplot(236), plt.imshow(fifth,'gray'), plt.title('6')
plt.show()

