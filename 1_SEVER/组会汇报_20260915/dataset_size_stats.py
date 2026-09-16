# -*- coding: utf-8 -*-
"""COCO small/medium/large instance statistics for citrus YOLO-seg datasets.

COCO thresholds on segmentation mask area:  small < 32^2,  medium 32^2~96^2, large >= 96^2 (px).
Two references: native image pixels, and 640-letterbox training input (matches the
existing dataset_difficulty audit convention).
"""
import os, glob, json
from PIL import Image

IMGSZ = 640
SMALL = 32 * 32          # 1024
LARGE = 96 * 96          # 9216

def poly_area(xs, ys):
    n = len(xs)
    a = 0.0
    for i in range(n):
        j = (i + 1) % n
        a += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(a) / 2.0

def cat(area):
    return 'small' if area < SMALL else ('medium' if area < LARGE else 'large')

def stats_for(root):
    out = {}
    img_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    for split in ('train', 'val', 'test'):
        lbldir = os.path.join(root, split, 'labels')
        imgdir = os.path.join(root, split, 'images')
        if not os.path.isdir(lbldir):
            continue
        counts = {'small': 0, 'medium': 0, 'large': 0}
        counts_nat = {'small': 0, 'medium': 0, 'large': 0}
        n_inst = 0
        n_img = 0
        min_sides = []
        areas640 = []
        for lf in sorted(glob.glob(os.path.join(lbldir, '*.txt'))):
            stem = os.path.splitext(os.path.basename(lf))[0]
            img = None
            for e in img_exts:
                p = os.path.join(imgdir, stem + e)
                if os.path.exists(p):
                    img = p
                    break
            if img is None:
                continue
            with Image.open(img) as im:
                W, H = im.size
            s = min(IMGSZ / W, IMGSZ / H)          # letterbox scale
            n_img += 1
            with open(lf) as f:
                for line in f:
                    t = line.split()
                    if len(t) < 5:
                        continue
                    v = [float(x) for x in t[1:]]
                    if len(v) == 4:                # bbox: cx cy w h
                        w_px, h_px = v[2] * W, v[3] * H
                        area_nat = w_px * h_px
                    else:                          # polygon
                        xs = v[0::2]; ys = v[1::2]
                        area_nat = poly_area(xs, ys) * W * H
                        xs_px = [x * W for x in xs]; ys_px = [y * H for y in ys]
                        w_px = max(xs_px) - min(xs_px); h_px = max(ys_px) - min(ys_px)
                    a640 = area_nat * s * s
                    n_inst += 1
                    areas640.append(a640)
                    min_sides.append(min(w_px, h_px) * s)
                    counts[cat(a640)] += 1
                    counts_nat[cat(area_nat)] += 1
        out[split] = {
            'images': n_img, 'instances': n_inst,
            'at_640': counts, 'at_native': counts_nat,
            'pct640': {k: round(100 * c / max(n_inst, 1), 2) for k, c in counts.items()},
            'min_side_lt8': sum(1 for m in min_sides if m < 8),
            'min_side_lt16': sum(1 for m in min_sides if m < 16),
            'area640_median': sorted(areas640)[len(areas640) // 2] if areas640 else 0,
        }
    return out

for root in [r'E:\mastercode\data\orange_yolo',
             r'E:\mastercode\data\orange_yolo_grouped_dedup_20260820']:
    print('=' * 70)
    print(root)
    r = stats_for(root)
    tot = {'images': 0, 'instances': 0,
           'at_640': {'small': 0, 'medium': 0, 'large': 0},
           'at_native': {'small': 0, 'medium': 0, 'large': 0}}
    for split, d in r.items():
        print(f"  {split:5s} imgs={d['images']:4d} inst={d['instances']:5d} | "
              f"@640 S/M/L = {d['at_640']['small']}/{d['at_640']['medium']}/{d['at_640']['large']} "
              f"({d['pct640']['small']}%/{d['pct640']['medium']}%/{d['pct640']['large']}%) | "
              f"native S/M/L = {d['at_native']['small']}/{d['at_native']['medium']}/{d['at_native']['large']} | "
              f"minSide<8:{d['min_side_lt8']} <16:{d['min_side_lt16']} | medArea640={d['area640_median']}")
        tot['images'] += d['images']; tot['instances'] += d['instances']
        for k in ('small', 'medium', 'large'):
            tot['at_640'][k] += d['at_640'][k]; tot['at_native'][k] += d['at_native'][k]
    n = tot['instances']
    print(f"  TOTAL imgs={tot['images']} inst={n} | @640 S/M/L = "
          f"{tot['at_640']['small']}/{tot['at_640']['medium']}/{tot['at_640']['large']} "
          f"({100*tot['at_640']['small']/n:.2f}%/{100*tot['at_640']['medium']/n:.2f}%/{100*tot['at_640']['large']/n:.2f}%) | "
          f"native S/M/L = {tot['at_native']['small']}/{tot['at_native']['medium']}/{tot['at_native']['large']}")
