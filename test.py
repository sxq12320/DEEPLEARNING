import torch
import cv2 
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r"E:\mastercode\data\Apple_RGB_D_Amoal\apple_rgbd_amoal_png\train\images\_MG_2821_04.png" , cv2.IMREAD_UNCHANGED)

print(img.shape) # type: ignore
img_depth = img[:,:,3] # type: ignore
img_bgr = img[:,:,:3] # type: ignore
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) 
print(f"--- 深度图全局统计 ---")
print(f"数据类型: {img_rgb.dtype}")
print(f"最大值: {img_rgb.max()} | 最小值: {img_rgb.min()}")
print(f"非零像素占比: {np.count_nonzero(img_rgb) / img_rgb.size * 100:.2f}%")

# B. 打印图像中心区域 (假设苹果在画面中央，截取 100x100 的局部)
# 这将直接向您展示非零的真实深度数值
h, w = img_depth.shape
center_h, center_w = h // 2, w // 2
print(f"\n--- 深度图中心区域 ({center_h-100}:{center_h+100}, {center_w-100}:{center_w+100}) ---")
print(img_rgb[center_h-100:center_h+100, center_w-100:center_w+100])

plt.subplot(1 , 2, 1)
plt.imshow(img_depth, cmap="gray" )
plt.subplot(1 , 2, 2)
plt.imshow(img_rgb  , cmap="gray")
plt.show()