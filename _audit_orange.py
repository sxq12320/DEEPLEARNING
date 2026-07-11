# -*- coding: utf-8 -*-
"""One-off audit of the orange_wuxi LabelMe dataset. Safe to delete."""
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(r"E:\mastercode\data\orange_wuxi")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}
BATCHES = [("annotions_x", "img"), ("annotion_x_2", "img_2")]


def stems(d, exts=None):
    p = ROOT / d
    if not p.exists():
        return {}
    out = {}
    for f in p.iterdir():
        if exts is None:
            if f.suffix.lower() == ".json":
                out[f.stem] = f
        elif f.suffix in exts or f.suffix.lower() in {e.lower() for e in exts}:
            out[f.stem] = f
    return out


grand_labels = Counter()
grand_shape_types = Counter()
grand_instances = 0
grand_imgs_with_ann = 0
lt6_polys = 0
empty_jsons = []
oob_coords = 0
embedded_imagedata = 0
per_img_instances = []
size_counter = Counter()
bad_label_examples = defaultdict(list)

print("=" * 70)
for ann_dir, img_dir in BATCHES:
    jstems = stems(ann_dir)
    istems = stems(img_dir, IMG_EXTS)
    if not jstems and not istems:
        print(f"[{ann_dir} / {img_dir}]  (absent)")
        continue
    only_json = sorted(set(jstems) - set(istems))
    only_img = sorted(set(istems) - set(jstems))
    print(f"[{ann_dir} / {img_dir}]")
    print(f"  json={len(jstems)}  img={len(istems)}  matched={len(set(jstems)&set(istems))}")
    print(f"  json-without-image: {len(only_json)}  {only_json[:6]}")
    print(f"  image-without-json: {len(only_img)}  {only_img[:6]}")

    for stem, jf in jstems.items():
        try:
            d = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  !! JSON parse fail {jf.name}: {e}")
            continue
        if d.get("imageData"):
            embedded_imagedata += 1
        W = d.get("imageWidth") or 0
        H = d.get("imageHeight") or 0
        size_counter[(W, H)] += 1
        shapes = d.get("shapes", [])
        if not shapes:
            empty_jsons.append(jf.name)
        grand_imgs_with_ann += 1
        per_img_instances.append(len(shapes))
        for s in shapes:
            lbl = s.get("label", "<none>")
            st = s.get("shape_type", "<none>")
            grand_labels[lbl] += 1
            grand_shape_types[st] += 1
            grand_instances += 1
            pts = s.get("points", [])
            if st == "polygon" and len(pts) < 6:
                lt6_polys += 1
            if lbl != "orange_immature":
                if len(bad_label_examples[lbl]) < 3:
                    bad_label_examples[lbl].append(jf.name)
            for (x, y) in pts:
                if W and H and (x < 0 or y < 0 or x > W or y > H):
                    oob_coords += 1
                    break
    print()

print("=" * 70)
print("TOTALS across all batches")
print(f"  images with annotation : {grand_imgs_with_ann}")
print(f"  total instances        : {grand_instances}")
print(f"  labels                 : {dict(grand_labels)}")
print(f"  shape_types            : {dict(grand_shape_types)}")
print(f"  polygons with <6 pts (converter SKIPS these): {lt6_polys}")
print(f"  empty jsons (0 shapes) : {len(empty_jsons)}  {empty_jsons[:8]}")
print(f"  out-of-bounds polygons : {oob_coords}")
print(f"  jsons w/ embedded imageData (bloat): {embedded_imagedata}")
if grand_labels and set(grand_labels) - {"orange_immature"}:
    print(f"  !! NON-standard labels (won't convert): "
          f"{ {k:v for k,v in grand_labels.items() if k!='orange_immature'} }")
    for k, ex in bad_label_examples.items():
        print(f"       {k!r} e.g. {ex}")
if per_img_instances:
    pi = sorted(per_img_instances)
    n = len(pi)
    print(f"  instances/image  min={pi[0]} med={pi[n//2]} "
          f"mean={sum(pi)/n:.1f} max={pi[-1]}")
    print(f"  images with 1 instance: {sum(1 for x in pi if x==1)} ; "
          f">20 instances: {sum(1 for x in pi if x>20)}")
print(f"  image sizes (top 6)    : {size_counter.most_common(6)}")
print("=" * 70)
