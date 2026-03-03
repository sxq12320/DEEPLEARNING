import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('class2\img\pic02.jpg')
hsv = cv2.cvtColor(img , cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

if img is None:
    print("no")
else:
    print("yes")
    h , w = img.shape[:2]
    print(h , w)   

lower1 = np.array([5, 20, 30])    
upper1 = np.array([30, 200, 255])
lower2 = np.array([150, 10, 30])
upper2 = np.array([180, 200, 255])

mask1 = cv2.inRange(hsv, lower1, upper1)
mask2 = cv2.inRange(hsv, lower2, upper2)
skin_mask = cv2.bitwise_or(mask1, mask2)

kernel_small = np.ones((3, 3), np.uint8)
kernel_large = np.ones((3, 3), np.uint8)

skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel_large, iterations=1)

contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
head_mask = np.zeros((h, w), np.uint8)  

min_area = 1500 
max_area = h * w // 4
aspect_ratio_range = (0.6, 1.5)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if not (min_area < area < max_area):
        continue

    x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
    if h_cnt == 0: 
        continue
    aspect_ratio = w_cnt / h_cnt
    if not (aspect_ratio_range[0] < aspect_ratio < aspect_ratio_range[1]):
        continue

    cv2.drawContours(head_mask, [cnt], -1, 255, -1)

head_mask = cv2.dilate(head_mask, kernel_small, iterations=1)

_, binary = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
head_roi = cv2.bitwise_and(binary, binary, mask=head_mask)

# 
# binary_head = cv2.adaptiveThreshold(
#     head_roi, 255,
#     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#     cv2.THRESH_BINARY_INV,
#     9, 3  
# )

result = np.zeros_like(img)  

result[:, :, 0] = head_roi
result[:, :, 1] = head_roi
result[:, :, 2] = head_roi

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(18, 8))
        
plt.subplot(121)
plt.imshow(img_rgb)
plt.title("origin picture")
plt.axis("off")

# plt.subplot(222)
# plt.imshow(hsv,cmap='hsv')
# plt.title("hsv")
# plt.axis("off")

# plt.subplot(223)
# plt.imshow(head_mask, cmap="gray")
# plt.title("head mask")
# plt.axis("off")

plt.subplot(122)
plt.imshow(result, cmap="gray")
plt.title("result")
plt.axis("off")

plt.tight_layout()
plt.show()

cv2.imwrite('pico2_result.jpg',result)