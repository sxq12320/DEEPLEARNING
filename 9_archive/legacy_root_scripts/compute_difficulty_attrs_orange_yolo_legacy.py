"""Compute difficulty attributes for orange_yolo dataset — one-shot, no deps beyond cv2+numpy."""
from pathlib import Path
import json, time, re
import cv2
import numpy as np

DATA = Path(r"E:/mastercode/data/orange_yolo")
IMGSZ = 640
RING = 16
BINS = 32  # for RGB histogram chi2

def load_labels(p: Path):
    polys, valid_idx = [], []
    for line in p.read_text(encoding="utf-8").splitlines():
        v = line.split()
        if len(v) < 7: continue
        xy = np.array(v[1:], dtype=float).reshape(-1, 2)
        polys.append(xy)
    return polys

def area_640(poly_norm, w, h):
    scale = IMGSZ / max(w, h)
    pts = poly_norm * np.array([w*scale, h*scale])
    x, y = pts[:,0], pts[:,1]
    return 0.5 * abs(np.dot(x, np.roll(y,1)) - np.dot(y, np.roll(x,1)))

def p2seg_dist(p, a, b):
    ab, ap = b - a, p - a
    t = np.clip(np.dot(ap,ab)/(np.dot(ab,ab)+1e-12), 0.0, 1.0)
    return float(np.linalg.norm(ap - t*ab))

def min_poly_dist(p1, p2):
    best = float("inf")
    for i in range(len(p1)):
        p = p1[i]
        for j in range(len(p2)):
            a, b = p2[j], p2[(j+1)%len(p2)]
            d = p2seg_dist(p,a,b)
            if d < best: best = d
    for j in range(len(p2)):
        p = p2[j]
        for i in range(len(p1)):
            a, b = p1[i], p1[(i+1)%len(p1)]
            d = p2seg_dist(p,a,b)
            if d < best: best = d
    return best

def poly_mask(poly_norm, w, h):
    pts = (poly_norm * np.array([w,h])).astype(np.int32)
    m = np.zeros((h,w), dtype=np.uint8)
    cv2.fillPoly(m, [pts], 1)
    return m

def lab_mean(img_bgr, mask):
    if mask.sum() < 10: return None
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(float)
    return lab[mask>0].mean(axis=0)

def delta_e(l1, l2):
    return float(np.linalg.norm(l1-l2)) if (l1 is not None and l2 is not None) else None

def chi2_hist(img_bgr, m_fg, m_ring):
    if m_fg.sum()<10 or m_ring.sum()<10: return None
    c = 0.0
    for ch in range(3):
        hf = np.histogram(img_bgr[m_fg>0,ch], bins=BINS, range=(0,256))[0].astype(float)
        hr = np.histogram(img_bgr[m_ring>0,ch], bins=BINS, range=(0,256))[0].astype(float)
        c += np.sum((hf-hr)**2/(hf+hr+1e-10))
    return float(c/3.0)

out = {"config": {"imgsz": IMGSZ, "ring_px": RING, "hist_bins": BINS, "date": "2026-07-13"}, "splits": {}}
all_inst = []
t0 = time.time()

for split in ["train","val","test"]:
    img_dir = DATA / split / "images"
    lbl_dir = DATA / split / "labels"
    si = {"images": {}, "instances": []}
    n_proc = 0
    for img_path in sorted(img_dir.glob("*.jpg")):
        stem = img_path.stem
        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w = img.shape[:2]
        polys_norm = load_labels(lbl_dir / f"{stem}.txt")
        polys_orig = [p * np.array([w,h]) for p in polys_norm]
        ni = len(polys_orig)
        si["images"][stem] = {"stem": stem, "w": w, "h": h, "n_inst": ni}
        min_dists = [float("inf")]*ni
        if ni >= 2:
            for i in range(ni):
                for j in range(i+1, ni):
                    d = min_poly_dist(polys_orig[i], polys_orig[j])
                    min_dists[i] = min(min_dists[i], d)
                    min_dists[j] = min(min_dists[j], d)
        for idx, (pn, po) in enumerate(zip(polys_norm, polys_orig)):
            a = area_640(pn, w, h)
            sm = "extreme" if a<256 else "small" if a<1024 else "medium" if a<4096 else "large"
            md = min_dists[idx]
            ad = "touching" if md<3 else "close" if md<RING else "separated" if md<float("inf") else "isolated"
            mf = poly_mask(pn, w, h)
            ksz = RING*2+1
            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz,ksz))
            mdil = cv2.dilate(mf, kern, iterations=1)
            mr = (mdil - mf).clip(0,1)
            lf = lab_mean(img, mf)
            lr = lab_mean(img, mr)
            de = delta_e(lf, lr)
            ch2 = chi2_hist(img, mf, mr)
            inst = {"image": stem, "idx": idx, "area_640": round(a,1), "small_level": sm,
                    "min_dist_px": round(md,2) if md<float("inf") else None, "adhesion_level": ad,
                    "delta_e": round(de,2) if de is not None else None, "rgb_chi2": round(ch2,4) if ch2 is not None else None}
            if de is not None and ch2 is not None:
                inst["camouflage_level"] = "high" if (de<8.0 and ch2<0.3) else "moderate" if (de<14.0 or ch2<0.6) else "low"
            else:
                inst["camouflage_level"] = "unknown"
            si["instances"].append(inst)
        n_proc += 1
        if n_proc % 200 == 0:
            print(f"  [{split}] {n_proc}... ({time.time()-t0:.0f}s)")
    out["splits"][split] = si
    all_inst.extend(si["instances"])
    print(f"  [{split}] done: {n_proc} imgs, {len(si['instances'])} instances ({time.time()-t0:.0f}s)")

# subsets
subsets = {}
for tag, attr, lvls in [("small","small_level",("extreme","small")),
                         ("dense","n_inst",("extreme","dense")),
                         ("adhesion","adhesion_level",("touching","close")),
                         ("camouflage","camouflage_level",("high","moderate"))]:
    s = set()
    if tag == "dense":
        for sd in out["splits"].values():
            for stem, info in sd["images"].items():
                if info.get("n_inst",0) >= 10:
                    s.add(stem)
    else:
        for inst in all_inst:
            if inst.get(attr,"") in lvls:
                s.add(inst["image"])
    subsets[tag] = sorted(s)

out["subsets"] = subsets
out["summary"] = {
    "total_images": sum(len(sd["images"]) for sd in out["splits"].values()),
    "total_instances": len(all_inst),
    "small_pct": round(sum(1 for i in all_inst if i["small_level"] in ("extreme","small"))/max(len(all_inst),1)*100, 1),
    "dense_imgs_pct": round(sum(1 for sd in out["splits"].values() for v in sd["images"].values() if v.get("n_inst",0)>=10)/max(sum(len(sd["images"]) for sd in out["splits"].values()),1)*100, 1),
    "adhesion_pct": round(sum(1 for i in all_inst if i["adhesion_level"] in ("touching","close"))/max(len(all_inst),1)*100, 1),
    "cam_high_pct": round(sum(1 for i in all_inst if i["camouflage_level"]=="high")/max(len(all_inst),1)*100, 1),
    "subset_sizes": {k: len(v) for k,v in subsets.items()},
}

dest = DATA / "difficulty_benchmark.json"
dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n{'='*60}")
for k,v in out["summary"].items():
    print(f"  {k}: {v}")
print(f"\nwritten -> {dest}")
print(f"{'='*60}")
