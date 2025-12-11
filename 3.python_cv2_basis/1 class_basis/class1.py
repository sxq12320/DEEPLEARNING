import cv2 as cv
import matplotlib as plt
import numpy as np


def cv_show(name,img):
    cv.imshow('name',img)
    cv.waitKey(0)
    cv.destroyAllWindows()

img=cv.imread('1.picture\cat.jpg')

img_shape=img.shape #读取图像的H W C 高度 宽度 彩色通道 BGR
#print(img_shape)

cv_show('hello',img)

#cv.imshow('image1',img)   #图像进行显示
#cv.waitKey(0)  #等待时间0表示任意键终止，若其他的数据是毫秒级的显示
#cv.destroyAllWindows()#释放窗口