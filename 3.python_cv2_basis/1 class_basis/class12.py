import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time

img = cv.imread('0.picture\pw.png')
img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
rat , img = cv.threshold(img,100,255,cv.THRESH_BINARY)
print(rat)

#礼帽： 原始输入-开运算结果
#黑帽： 闭运算-原始输入

kernel = np.ones((3,3),np.uint8)
tophat = cv.morphologyEx(img,cv.MORPH_TOPHAT,kernel)
blachat= cv.morphologyEx(img,cv.MORPH_BLACKHAT,kernel)

title = ['orginal picutre' ,'tophat','balckhat']
img = [img , tophat , blachat]

for i in range(0,3,1):
    plt.subplot(220+i+1)
    plt.imshow(img[i],'gray')
    plt.title(title[i])
plt.show()