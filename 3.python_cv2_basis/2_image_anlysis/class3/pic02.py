# -*- coding: utf-8 -*-
"""
K-Means 人像提取
仅依赖：cv2、numpy、matplotlib
无需 sys
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def kmeans_portrait(img_bgr, k=2, Lab_weight=(1, 1, 1)):
    """返回 mask(0/255) + 提取后人像（BGR）"""
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    # 加权
    wL, wA, wB = Lab_weight
    img_lab[:, :, 0] *= wL
    img_lab[:, :, 1] *= wA
    img_lab[:, :, 2] *= wB

    h, w = img_lab.shape[:2]
    samples = img_lab.reshape(-1, 3)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(samples, k, None, criteria, 10,
                                    cv2.KMEANS_RANDOM_CENTERS)

    # 最亮簇当背景
    bg_idx = np.argmax(centers[:, 0])
    fg_mask = (labels != bg_idx).astype(np.uint8) * 255
    fg_mask = fg_mask.reshape(h, w)

    portrait = cv2.bitwise_and(img_bgr, img_bgr, mask=fg_mask)
    return fg_mask, portrait

def main():
    # 1. 把图片路径直接写这里即可，不需要 sys
    img_path = 'class3\img\pic02.png'          # <-- 改成你的文件

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError('读图失败，请检查路径')

    mask, portrait = kmeans_portrait(img)

    # 2. 可视化
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('original')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(mask, cmap='gray')
    plt.title('mask')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()