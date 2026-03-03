import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def adjust_highlights_claahe(image , clip_limit = 2.0 , grid_size = (8,8)): 
    lab = cv.cvtColor(image , cv.COLOR_BGR2LAB)
    l , a , b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_clahe = clahe.apply(l)
    lab_clahe = cv.merge([l_clahe, a, b])
    return cv.cvtColor(lab_clahe, cv.COLOR_LAB2BGR)

# def gamma_correction(image , gamma = 1.0):
#     inv_gamma = 1.0/gamma
#     table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
#     return cv.LUT(image , table)

img = cv.imread('class1\img\pic06.tif')

Threshold = 120
reduce_factor = 0.4

img_hsv = cv.cvtColor(img , cv.COLOR_BGR2HSV)
h , s , v = cv.split(img_hsv)
mask = v > Threshold
v_reduce = np.where(mask , v * reduce_factor , v).astype(np.uint8)

hsv_reduced = cv.merge([h , s , v_reduce])
enhanced1 = cv.cvtColor(hsv_reduced , cv.COLOR_HSV2BGR)

enhanced2 =  adjust_highlights_claahe(img , clip_limit=5.0)

plt.subplot(1 , 3 , 1)
plt.imshow(img , cmap='gray')
plt.title('before')
plt.axis('off')

plt.subplot(1 , 3 , 2)
plt.imshow(enhanced1 , cmap='gray')
plt.title('after_first')
plt.axis('off')

plt.subplot(1 , 3 , 3)
plt.imshow(enhanced2 , cmap='gray')
plt.title('after_second')
plt.axis('off')

plt.tight_layout()
plt.show()
