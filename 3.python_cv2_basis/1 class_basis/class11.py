import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time

img = cv.imread('0.picture\pw.png')
img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
rat , img = cv.threshold(img,128,255,cv.THRESH_BINARY)

kernel = np.ones((5,5),np.uint8)
eroding = cv.erode(img , kernel,iterations=3)
dilating = cv.dilate(img , kernel , iterations=3)

gradient = cv.morphologyEx(img , cv.MORPH_GRADIENT , kernel)   #梯度计算

title = ['orginal picutre' , 'eroding' , 'dilating' , 'gradient']
img = [img , eroding , dilating , gradient]

for i in range(0,4,1):
    plt.subplot(220+i+1)
    plt.imshow(img[i],'gray')
    plt.title(title[i])
plt.show()