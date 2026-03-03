import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('class1/img/pic05.tif')


if len(img.shape)==3:
    img = cv.cvtColor(img , cv.COLOR_BGR2GRAY)

laplacian_kernel = np.array([
    [0 , -1 , 0],
    [-1 , 5 , -1],
    [0 , -1 , 0]
])
sharpened = cv.filter2D(img , -1 , laplacian_kernel)

t_otsu , enhanced = cv.threshold(sharpened , 0 , 255 , cv.THRESH_BINARY + cv.THRESH_OTSU)

plt.subplot(1 , 2 , 1)
plt.imshow(img , cmap='gray')
plt.title('before')
plt.axis('off')

plt.subplot(1 , 2 , 2)
plt.imshow(enhanced , cmap='gray')
plt.title('after')
plt.axis('off')

plt.tight_layout()
plt.show()