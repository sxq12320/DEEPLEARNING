import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('0.picture\cat.jpg')
print(img.shape)

img = img[0:50,0:200] #切割图片大小

cur_Img_R = img.copy()
cur_Img_R[:,:,0] = 0
cur_Img_R[:,:,1] = 0

cur_Img_G = img.copy()
cur_Img_G[:,:,0] = 0  
cur_Img_G[:,:,2] = 0

cur_Img_B = img.copy()
cur_Img_B[:,:,1] = 0
cur_Img_B[:,:,2] = 0

'''
plt.subplot(221),plt.imshow(cur_Img_R,'brg'),plt.title('1')
plt.subplot(222),plt.imshow(cur_Img_G,'brg'),plt.title('2')
plt.subplot(223),plt.imshow(cur_Img_B,'brg'),plt.title('3')
plt.subplot(224),plt.imshow(img,'brg'),plt.title('4')
plt.show()

'''
cv.imshow('R',cur_Img_R)
cv.imshow('G',cur_Img_G)
cv.imshow('B',cur_Img_B)


cv.waitKey(0)
cv.destroyAllWindows()


