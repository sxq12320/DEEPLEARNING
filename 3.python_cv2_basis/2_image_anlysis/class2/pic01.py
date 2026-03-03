import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('class2\img\pic01.tiff')

if img is None:
    print("no")
else:
    print("yes")

gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img , cv2.COLOR_BGR2HSV)

#G: H-35~70 S-70~255  V-40~255
lower_green = np.array([35 , 70 , 40])
upper_green = np.array([70 , 255 , 255])

#Y: H-20~35 S-100~255 V-120~255
lower_yellow = np.array([20 , 100 , 120])
upper_yellow = np.array([35 , 255 , 255])

mask_green = cv2.inRange(hsv , lower_green , upper_green)
mask_yellow = cv2.inRange(hsv , lower_yellow , upper_yellow)
mask_color = cv2.bitwise_or(mask_green , mask_yellow)

kernel_small = np.ones((2,2),np.uint8)

mask_color = cv2.dilate(mask_color , kernel_small , iterations=1)
mask_color = cv2.erode(mask_color , kernel_small , iterations=1)

mask_color = cv2.medianBlur(mask_color , 5)#中值滤波降噪

result = cv2.bitwise_and(img , img , mask=mask_color)

bgr_result = result.copy()
result = cv2.cvtColor(result , cv2.COLOR_BGR2RGB)


img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)

plt.subplot(1 , 3 , 1)
plt.imshow(img)
plt.title("origin picture")
plt.axis('off')

plt.subplot(1 , 3 , 2)
plt.imshow(mask_color,cmap='hsv')
plt.title("canny picture")
plt.axis('off')

plt.subplot(1 , 3 , 3)
plt.imshow(result)
plt.title("result picture")
plt.axis('off')

plt.show()

cv2.imwrite('pic01_result.jpg',bgr_result)