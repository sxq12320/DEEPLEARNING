import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time

#开 先erode 在dilate
#闭 先dilate 后erode

def cv_show(name,img):
    cv.imshow('name',img)
    cv.waitKey(0)
    cv.destroyAllWindows()

img = cv.imread('0.picture\pw.png')
img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
rat , img = cv.threshold(img,128,255,cv.THRESH_BINARY)

kernel = np.ones((5,5),np.uint8)
print(kernel)
opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)    #开 先erode 在dilate
closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)  #闭 先dilate 后erode

title = ['orginal pirture' , 'opening' , 'closing']
img = [img , opening , closing]

for i in range(0,3,1):
    key = 220+i+1
    plt.subplot(key)
    plt.imshow(img[i],'gray')
    plt.title(title[i])
plt.show()