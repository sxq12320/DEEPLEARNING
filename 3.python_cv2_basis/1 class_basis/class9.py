import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time

img0 = cv.imread('0.picture\pw.png')
print(img0.shape)
img0 = img0[0:928,130:1000,:]
img = cv.cvtColor(img0,cv.COLOR_BGR2GRAY)
rat,img = cv.threshold(img,90,255,cv.THRESH_BINARY)

print(img)


kernel1 = np.ones((2,2),np.uint8)
res = cv.erode(img,kernel1,iterations=4)#腐蚀操作

fina = cv.dilate(img,kernel1,iterations=1)#膨胀


plt.subplot(231),plt.imshow(img0,'gray'),plt.title('orginal picture')
plt.subplot(232),plt.imshow(img,'gray'),plt.title('orginal picture')
plt.subplot(233),plt.imshow(res,'gray'),plt.title('finally')
plt.subplot(234),plt.imshow(fina,'gray'),plt.title('pengzhang')
plt.show()