import cv2 as cv
import matplotlib as plt
import numpy as np

def cv_show(name,img):
    cv.imshow('name',img)
    cv.waitKey(0)
    cv.destroyAllWindows()


Img_Gray = cv.imread('0.picture\cat.jpg',cv.IMREAD_GRAYSCALE) #读取灰度图片
print(Img_Gray)

cv.imshow('img',Img_Gray)
cv.waitKey(0)
cv.destroyAllWindows()

cv.imwrite('Img_cat.png',Img_Gray) #保存图像

Img_Gray_shape = Img_Gray.shape
print(Img_Gray_shape) 

