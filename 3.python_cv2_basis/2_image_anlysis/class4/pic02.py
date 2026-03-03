import numpy as np
import cv2 
import matplotlib.pyplot as plt
'''
共四个步骤
1.二值化处理
2.利用闭运算去掉小黑球
3.利用开运算去掉打黑求旁边的白色背景
4.边缘检测
'''

image = cv2.imread("class4\img\pic02.tif")
_,binary = cv2.threshold(image , 100 , 255 , cv2.THRESH_BINARY)

Kernel_small_circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE , (60,60))
closing = cv2.morphologyEx(binary , cv2.MORPH_CLOSE , Kernel_small_circle)

Kernel_big_circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE , (150,150))
opening = cv2.morphologyEx(closing , cv2.MORPH_OPEN , Kernel_big_circle)

edges = cv2.Canny(opening , 50 , 150)
res = image.copy()
res[edges != 0]=[0 , 0 , 0]

plt.subplot(221)
plt.imshow(binary , cmap='gray')
plt.title("binary picture")
plt.axis('off')

plt.subplot(222)
plt.imshow(closing , cmap='gray')
plt.title("only big circle")
plt.axis('off')

plt.subplot(223)
plt.imshow(opening , cmap='gray')
plt.title("big circle background")
plt.axis('off')

plt.subplot(224)
plt.imshow(res , cmap='gray')
plt.title("result")
plt.axis('off')
plt.show()

cv2.imwrite("pic02_res.jpg",res)