import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('class2\img\pic03.jpg')
hsv = cv2.cvtColor(img , cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

if img is None:
    print("no")
else:
    print("yes")
    h , w = img.shape[:2]
    print(h , w)   

lower1 = np.array([00, 30, 80])
upper1 = np.array([20, 170, 255])
lower2 = np.array([160, 10, 80])
upper2 = np.array([180, 170, 255])
skin_mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))

kernel_small = np.ones((3, 3), np.uint8)
kernel_large = np.ones((3, 3), np.uint8)

skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel_large, iterations=1)
skin_mask = cv2.medianBlur(skin_mask , 3)

skin_roi = cv2.bitwise_and(gray, gray, mask=skin_mask)
_, binary = cv2.threshold(skin_roi, 180, 255, cv2.THRESH_BINARY)


result = np.zeros_like(img)  

result[:, :, 0] = binary
result[:, :, 1] = binary
result[:, :, 2] = binary
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(18, 6))

        
plt.subplot(121)
plt.imshow(img_rgb)
plt.title("origin picture")
plt.axis("off")

plt.subplot(122)
plt.imshow(result, cmap="gray")
plt.title("result")
plt.axis("off")

plt.tight_layout()
plt.show()

cv2.imwrite('pico3_result.jpg',result)