import cv2
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
import shutil
from PIL import Image


VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
               'dog', 'horse', 'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

def mask_to_polygon(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 10:
            continue
        epsilon = 0.002 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True).reshape(-1, 2)
        if len(approx) >= 3:
            polygons.append(approx)
    return polygons

def convert(voc_root, split='train'):
    voc_root = Path(voc_root)
    split_file = voc_root / 'ImageSets' / 'Segmentation' / f'{split}.txt'
    img_ids = [x.strip() for x in split_file.read_text().strip().split('\n')]

    out_img_dir = Path(f'yolo_voc/images/{split}')
    out_lbl_dir = Path(f'yolo_voc/labels/{split}')
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    converted, skipped = 0, 0
    for img_id in img_ids:
        inst_mask_path = voc_root / 'SegmentationObject' / f'{img_id}.png'
        xml_path = voc_root / 'Annotations' / f'{img_id}.xml'
        img_path = voc_root / 'JPEGImages' / f'{img_id}.jpg'

        if not inst_mask_path.exists() or not xml_path.exists():
            skipped += 1
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()
        w = int(root.find('size/width').text)
        h = int(root.find('size/height').text)

        # 读取调色板PNG，像素值即实例编号
        # inst_mask = cv2.imread(str(inst_mask_path), cv2.IMREAD_GRAYSCALE)
        inst_mask = np.array(Image.open(str(inst_mask_path)))

        lines = []
        for idx, obj in enumerate(root.findall('object'), start=1):
            cls_name = obj.find('name').text
            if cls_name not in VOC_CLASSES:
                continue
            cls_id = VOC_CLASSES.index(cls_name)

            binary = (inst_mask == idx).astype(np.uint8) * 255
            if binary.sum() == 0:
                continue

            for poly in mask_to_polygon(binary):
                coords = [f"{pt[0]/w:.6f} {pt[1]/h:.6f}" for pt in poly]
                lines.append(f"{cls_id} " + " ".join(coords))

        if lines:
            (out_lbl_dir / f'{img_id}.txt').write_text('\n'.join(lines))
            shutil.copy(img_path, out_img_dir / f'{img_id}.jpg')
            converted += 1

    print(f'{split}: 转换 {converted} 张，跳过 {skipped} 张')

# 修改为你的实际路径
TRAINVAL = r'E:\mastercode\data\VOC\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007'
TEST     = r'E:\mastercode\data\VOC\VOCtest_06-Nov-2007\VOCdevkit\VOC2007'

voc_root = Path(TRAINVAL)
val_ids = (voc_root / 'ImageSets/Segmentation/val.txt').read_text().strip().split('\n')
missing = [i for i in val_ids if not (voc_root / 'SegmentationObject' / f'{i.strip()}.png').exists()]
print(f'val 共 {len(val_ids)} 张，缺少 mask: {len(missing)} 张')

convert(TRAINVAL, 'train')
convert(TRAINVAL, 'val')
convert(TEST, 'test')