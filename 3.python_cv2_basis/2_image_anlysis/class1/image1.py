import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('class1\img\image1-FG.jpg')
img_hsv = cv.cvtColor(img , cv.COLOR_BGR2HSV)
img_bg = cv.imread('class1\img\image1-BG.jpg')
bg =  cv.imread('class1\img\image1-BG.jpg')


mask_lower = np.array([30,200,100])
mask_upper = np.array([80,260,140])

mask = cv.inRange(img_hsv,mask_lower,mask_upper)
green = cv.bitwise_and(img , img , mask=mask)
gray_green = cv.cvtColor(green , cv.COLOR_BGR2GRAY)
_ , black_green = cv.threshold(gray_green , 0 , 200 , cv.THRESH_BINARY)
Dinosaur = img - green # 扣出恐龙

h , w = img_bg.shape[:2]
fg = black_green.copy()
fg = cv.resize(fg , (w,h),interpolation = cv.INTER_NEAREST)
black_mask = fg==0
img_bg[black_mask]=[0,0,0]#黑色区域处理好了

Dinosaur = cv.resize(Dinosaur , (w,h),interpolation = cv.INTER_NEAREST)
result = Dinosaur + img_bg


plt.subplot(1,3,1)
plt.imshow(img,cmap='gray')
plt.title('before')
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(bg,cmap='gray')
plt.title('before')
plt.axis('off')


plt.subplot(1,3,3)
plt.imshow(result , cmap='gray')
plt.title('result')
plt.axis('off')

plt.tight_layout()
plt.show()