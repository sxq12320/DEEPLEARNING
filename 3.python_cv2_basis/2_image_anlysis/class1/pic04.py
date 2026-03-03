import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('class1\img\pic04.jpg')

img_float = img.astype(np.float32) / 255.0

# BGR
img_float[:, :, 0] *= 0.9   # 蓝色通道
img_float[:, :, 1] *= 5.0   # 绿色通道
img_float[:, :, 2] *= 0.9   # 红色通道

img_corrected = np.clip(img_float*255, 0, 255).astype(np.uint8)

plt.subplot(1, 2, 1)
plt.title('Original')
plt.imshow(img)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('after')
plt.imshow(img_corrected)
plt.axis('off')

plt.tight_layout()
plt.show()