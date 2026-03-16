from ultralytics import SAM
import cv2
import os
import numpy as np

if __name__ == '__main__':
    sam = SAM(r'E:\mastercode\6.yolo\sam2.1_t.pt')

    img_dir = r"E:\mastercode\data\VOC\yolo_voc\images\test"
    img_path = os.path.join(img_dir, os.listdir(img_dir)[2])  # 换几张试试
    print(f"测试图片: {img_path}")

    # SAM2 全图自动分割
    results = sam(img_path)

    img = cv2.imread(img_path)
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        print(f"分割出 {len(masks)} 个实例")

        for mask in masks:
            color = np.random.randint(0, 255, 3).tolist()
            img[mask.astype(bool)] = (
                img[mask.astype(bool)] * 0.4 + np.array(color) * 0.6
            ).astype(np.uint8)
    else:
        print("未分割出任何目标")

    out_path = r"E:\mastercode\6.yolo\output_sam2.jpg"
    cv2.imwrite(out_path, img)
    print(f"结果保存: {out_path}")