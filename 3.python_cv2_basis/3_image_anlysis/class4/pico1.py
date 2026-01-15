import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
共四个步骤
1.利用开运算将横线去掉
2.利用开运算去掉竖线
3.利用开运算将小字母去掉
4.做减法
'''

image = cv2.imread("class4\img\pic01.tif")
h , w = image.shape[:2]
upper_height = h//3
img = image.copy()
img[0:upper_height,0:w] = (0,0,0)
result=img.copy()

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

kernel_noise = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  #小核去除细碎噪点
img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_noise)

kernel_horitoral = cv2.getStructuringElement(cv2.MORPH_RECT , (30,1))
opening_h = cv2.morphologyEx(img , cv2.MORPH_OPEN , kernel_horitoral)
closing_h = cv2.morphologyEx(opening_h , cv2.MORPH_CLOSE , kernel_horitoral)

kernel_vterial = cv2.getStructuringElement(cv2.MORPH_RECT , (1,30))
opening_v = cv2.morphologyEx(img , cv2.MORPH_OPEN , kernel_vterial)
closing_v = cv2.morphologyEx(opening_v , cv2.MORPH_CLOSE , kernel_vterial)

all = cv2.max(closing_h , closing_v)
res = cv2.subtract(result , all)

kernel_repair = cv2.getStructuringElement(cv2.MORPH_RECT , (2,2))
res1 = cv2.morphologyEx(res , cv2.MORPH_CLOSE , kernel_repair)

plt.subplot(221)
plt.imshow(image , cmap='gray')
plt.title("binary picture")
plt.axis('off')

plt.subplot(222)
plt.imshow(closing_h , cmap='gray')
plt.title("only big circle")
plt.axis('off')

plt.subplot(223)
plt.imshow(closing_v  , cmap='gray')
plt.title("big circle background")
plt.axis('off')

plt.subplot(224)
plt.imshow(res1 , cmap='gray')
plt.title("result")
plt.axis('off')
plt.show()

cv2.imwrite("pic02_result.jpg",res1)