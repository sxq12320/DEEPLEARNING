import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img_cat = cv.imread('0.picture\cat2.jpg')
img_dog = cv.imread('0.picture\dog2.jpg')

img_dog_shape = img_dog.shape
img_cat_shape = img_cat.shape
print(img_cat_shape)
print(img_dog_shape)

img_dog = cv.resize(img_dog,(1390,770))
img_cat = cv.resize(img_cat,(1390,770))
#resize 超级坑人

print(img_cat_shape)
print(img_dog_shape)

res = cv.addWeighted(img_dog,0.5,img_cat,0.5,0)


plt.subplot(221),plt.imshow(img_cat,'flag'),plt.title('first')
plt.subplot(222),plt.imshow(img_dog,'gray'),plt.title('third')
plt.subplot(223),plt.imshow(res,'gray'),plt.title('forth')
plt.show()
