import cv2 
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("class3\img\pic01.jpg")
img_hsv = cv2.cvtColor(img , cv2.COLOR_BGR2HSV)
img_rgb = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)

h,s,v = cv2.split(img_hsv)
yellow_mask = (h>=10) & (h <= 90)
h[yellow_mask] = h[yellow_mask]*0.5
adjusted_hsv = cv2.merge([h , s , v])

pixels = adjusted_hsv.reshape((-1 , 3)).astype(np.float32)

criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER , 100 , 0.1)
k =2#K means进行两个颜色的分割

_ , labels , centers = cv2.kmeans(pixels , k , None , criteria , 10 , cv2.KMEANS_PP_CENTERS)

segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(img.shape).astype(np.uint8)
#将花朵和其他部分进行了分割
labels_reshaped = labels.reshape(img.shape[:2])

target_label =1  

# 创建掩码：目标标签区域为1，其他区域为0
mask = np.where(labels_reshaped == target_label, 1, 0).astype(np.uint8)
kernel = np.ones((3,3),np.uint8)
mask = cv2.morphologyEx(mask , cv2.MORPH_OPEN , kernel , iterations=5)

# 将掩码扩展为3通道，与原图匹配
mask_3d = np.stack([mask, mask, mask], axis=-1)

# 应用掩码：目标区域保留原图颜色，其他区域变黑
extracted_object = img_rgb * mask_3d

plt.imshow(extracted_object)
plt.show()