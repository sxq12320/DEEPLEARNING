import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('class1\img\pic03.tif')

if len(img.shape)==3:
    img = cv.cvtColor(img , cv.COLOR_BGR2GRAY)

sharpen_basic_x = np.array([
    [-1 , 0 , -1],
    [-2 , 0 , 2],
    [-1 , 0 , 1]
])#sober_x

sharpen_basic_y = np.array([
    [-1 , -2 , -1],
    [0 , 0 , 0],
    [1 , 2 , 1]
])#sober_y

edge_x = cv.filter2D(img , -1 , sharpen_basic_x) # 卷积操作 
edge_y = cv.filter2D(img , -1 , sharpen_basic_y) # 卷积操作 

edge_combline = cv.addWeighted(
    cv.convertScaleAbs(edge_x) , 0.5,
    cv.convertScaleAbs(edge_y) , 0.5,
    0
)

sharpen_img = cv.addWeighted(img , 1.0 , edge_combline , 1.0 , 0)

plt.subplot(1,2,1)
plt.imshow(img , cmap= 'gray')
plt.title('before')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(sharpen_img,cmap= 'gray')
plt.title('after')
plt.axis('off')

plt.tight_layout()
plt.show()
print(edge_x)