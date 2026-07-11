"""Offline data augmentation for the orange_wuxi YOLO-seg dataset.

Generates augmented COPIES (real .jpg + .txt files) of the TRAIN split only.
val/ and test/ are never read or written — evaluation stays on clean data.

Design
------
* Geometry (image + polygons) is a single 3x3 homography (scale / flip / rotate /
  shear / translate / mild perspective). Polygon vertices are transformed directly
  and clipped to the frame with Sutherland-Hodgman — exact, and no mask raster
  round-trip. Out-of-frame area is padded with YOLO grey (114) so we never create
  UNLABELLED phantom fruit (which mirror/reflect padding would).
* Photometry (image only) is tuned for green-immature citrus on green foliage:
  brightness/contrast, MODEST hue jitter (keep the green-vs-green cue), saturation/
  value, one-of blur (camera/robot motion), Gaussian noise, and a soft cast shadow.
* Negatives (empty label) are augmented too and stay empty.

Multi-image augmentations (mosaic / mixup / copy_paste) are intentionally left to
YOLO's online pipeline — this script is per-image only.

Run (test a few with overlays first, then full):
    python augment_orange_offline.py --limit 5 --n-aug 2 --debug-overlay 3
    python augment_orange_offline.py --clean --n-aug 3
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import cv2
import numpy as np

PAD = 114  # YOLO letterbox grey; fills out-of-frame regions (no phantom fruit)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline polygon-correct augmentation of the TRAIN split.")
    p.add_argument("--data-root", default="E:/mastercode/data/test", help="YOLO dataset root (has images/ labels/).")
    p.add_argument("--n-aug", type=int, default=3, help="Augmented copies per train image.")
    p.add_argument("--seed", type=int, default=20260711)
    p.add_argument("--limit", type=int, default=0, help="Process only the first N train images (0 = all).")
    p.add_argument("--min-visible-frac", type=float, default=0.10,
                   help="Drop an instance if the clipped polygon keeps < this fraction of its original area.")
    p.add_argument("--out-long-side", type=int, default=0,
                   help="Downscale output so its long side = this many px (0 = keep native resolution).")
    p.add_argument("--jpg-quality", type=int, default=95)
    p.add_argument("--max-points", type=int, default=64, help="Cap polygon vertices (approxPolyDP simplification).")
    p.add_argument("--clean", action="store_true", help="Delete existing *_aug* train images/labels before running.")
    p.add_argument("--debug-overlay", type=int, default=0,
                   help="Save N side-by-side polygon overlays (orig | aug) to <root>/_aug_debug/ for eyeballing.")
    return p.parse_args()


# ----------------------------------------------------------------------------- IO
def read_label(path: Path) -> list[tuple[int, np.ndarray]]:
    """Parse a YOLO-seg label -> [(cls, poly_norm (N,2) in [0,1]), ...]. Missing/empty -> []."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7:  # cls + >=3 xy pairs
            continue
        cls = int(float(parts[0]))
        coords = np.array(parts[1:], dtype=np.float64)
        coords = coords[: (len(coords) // 2) * 2].reshape(-1, 2)
        out.append((cls, coords))
    return out


def write_label(path: Path, insts: list[tuple[int, np.ndarray]], w: int, h: int) -> None:
    lines = []
    for cls, poly_px in insts:
        norm = poly_px.copy().astype(np.float64)
        norm[:, 0] = np.clip(norm[:, 0] / w, 0.0, 1.0)
        norm[:, 1] = np.clip(norm[:, 1] / h, 0.0, 1.0)
        flat = " ".join(f"{v:.6f}" for v in norm.reshape(-1))
        lines.append(f"{cls} {flat}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# ----------------------------------------------------------------- geometry
def random_homography(w: int, h: int, rng: random.Random) -> np.ndarray:
    """Compose scale / x-flip / shear / rotate about the image centre + translate + mild perspective."""
    cx, cy = w / 2.0, h / 2.0
    scale = rng.uniform(0.6, 1.35)              # zoom out/in -> strong scale diversity
    angle = math.radians(rng.uniform(-15, 15))  # camera / branch tilt
    shear = math.radians(rng.uniform(-6, 6))
    flip = -1.0 if rng.random() < 0.5 else 1.0  # horizontal flip
    tx = rng.uniform(-0.08, 0.08) * w
    ty = rng.uniform(-0.08, 0.08) * h

    to_origin = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
    S = np.array([[scale * flip, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64)
    Sh = np.array([[1, math.tan(shear), 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    ca, sa = math.cos(angle), math.sin(angle)
    R = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=np.float64)
    back = np.array([[1, 0, cx + tx], [0, 1, cy + ty], [0, 0, 1]], dtype=np.float64)
    M = back @ R @ Sh @ S @ to_origin

    if rng.random() < 0.35:  # mild perspective, keyed off the frame corners
        m = 0.0008
        persp = np.array([[1, 0, 0], [0, 1, 0],
                          [rng.uniform(-m, m) / w * w, rng.uniform(-m, m) / h * h, 1]], dtype=np.float64)
        # perturb the projective row directly (small)
        persp[2, 0] = rng.uniform(-m, m)
        persp[2, 1] = rng.uniform(-m, m)
        M = persp @ M
    return M


def transform_points(poly_px: np.ndarray, M: np.ndarray) -> np.ndarray:
    ones = np.ones((poly_px.shape[0], 1))
    hom = np.hstack([poly_px, ones]) @ M.T
    w = hom[:, 2:3]
    w[np.abs(w) < 1e-9] = 1e-9
    return hom[:, :2] / w


def clip_to_rect(poly: np.ndarray, w: int, h: int) -> np.ndarray:
    """Sutherland-Hodgman clip of a (possibly non-convex) polygon to [0,w]x[0,h]."""
    def clip(pts, keep, inter):
        res = []
        n = len(pts)
        for i in range(n):
            cur, prv = pts[i], pts[i - 1]
            cin, pin = keep(cur), keep(prv)
            if cin:
                if not pin:
                    res.append(inter(prv, cur))
                res.append(cur)
            elif pin:
                res.append(inter(prv, cur))
        return res

    def isect(a, b, axis, val):
        ax, ay = a; bx, by = b
        if axis == "x":
            t = (val - ax) / ((bx - ax) if (bx - ax) != 0 else 1e-9)
            return (val, ay + t * (by - ay))
        t = (val - ay) / ((by - ay) if (by - ay) != 0 else 1e-9)
        return (ax + t * (bx - ax), val)

    pts = [tuple(p) for p in poly]
    for keep, inter in (
        (lambda p: p[0] >= 0, lambda a, b: isect(a, b, "x", 0)),
        (lambda p: p[0] <= w, lambda a, b: isect(a, b, "x", w)),
        (lambda p: p[1] >= 0, lambda a, b: isect(a, b, "y", 0)),
        (lambda p: p[1] <= h, lambda a, b: isect(a, b, "y", h)),
    ):
        if not pts:
            break
        pts = clip(pts, keep, inter)
    return np.array(pts, dtype=np.float64) if pts else np.empty((0, 2))


def poly_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def simplify(poly: np.ndarray, max_pts: int) -> np.ndarray:
    if len(poly) <= max_pts:
        return poly
    peri = cv2.arcLength(poly.astype(np.float32), True)
    for eps in (0.003, 0.005, 0.008, 0.012, 0.02):
        approx = cv2.approxPolyDP(poly.astype(np.float32), eps * peri, True).reshape(-1, 2)
        if len(approx) <= max_pts:
            return approx.astype(np.float64) if len(approx) >= 3 else poly
    return poly


# --------------------------------------------------------------- photometry
def photometric(img: np.ndarray, rng: random.Random) -> np.ndarray:
    out = img.astype(np.float32)

    # HSV jitter — modest hue to preserve the green-fruit vs green-leaf cue
    hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-6, 6)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.7, 1.3), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.6, 1.4), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    # brightness / contrast
    out = np.clip(out * rng.uniform(0.8, 1.2) + rng.uniform(-18, 18), 0, 255)

    # one-of blur (camera / robot-arm motion, defocus)
    r = rng.random()
    if r < 0.20:
        k = rng.choice([3, 5, 7])
        out = cv2.GaussianBlur(out, (k, k), 0)
    elif r < 0.35:  # linear motion blur
        k = rng.choice([9, 13, 17])
        kern = np.zeros((k, k), np.float32)
        kern[k // 2, :] = 1.0 / k
        ang = rng.uniform(0, 180)
        kern = cv2.warpAffine(kern, cv2.getRotationMatrix2D((k / 2, k / 2), ang, 1.0), (k, k))
        out = cv2.filter2D(out, -1, kern)

    # sensor noise
    if rng.random() < 0.30:
        out = np.clip(out + rng.uniform(4, 16) * np.random.randn(*out.shape).astype(np.float32), 0, 255)

    # soft cast shadow (branch/leaf) — a darkened random quad, blurred
    if rng.random() < 0.20:
        h, w = out.shape[:2]
        mask = np.zeros((h, w), np.float32)
        quad = np.array([[rng.randint(0, w), rng.randint(0, h)] for _ in range(rng.choice([3, 4]))], np.int32)
        cv2.fillPoly(mask, [quad], 1.0)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(w, h) * 0.02)
        factor = 1.0 - mask[..., None] * rng.uniform(0.25, 0.5)
        out = np.clip(out * factor, 0, 255)

    return out.astype(np.uint8)


# --------------------------------------------------------------- per image
def augment_one(img, insts, rng, args):
    h, w = img.shape[:2]
    M = random_homography(w, h, rng)
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(PAD, PAD, PAD))
    new_insts = []
    for cls, poly_norm in insts:
        poly_px = poly_norm.copy()
        poly_px[:, 0] *= w
        poly_px[:, 1] *= h
        orig_area = poly_area(poly_px)
        tp = transform_points(poly_px, M)
        cp = clip_to_rect(tp, w, h)
        if len(cp) < 3:
            continue
        if orig_area > 0 and poly_area(cp) < args.min_visible_frac * orig_area:
            continue
        if poly_area(cp) < 16:  # absolute sliver guard (px^2)
            continue
        new_insts.append((cls, simplify(cp, args.max_points)))

    out_img = photometric(warped, rng)

    if args.out_long_side and max(h, w) > args.out_long_side:
        s = args.out_long_side / max(h, w)
        nw, nh = int(round(w * s)), int(round(h * s))
        out_img = cv2.resize(out_img, (nw, nh), interpolation=cv2.INTER_AREA)
        for i, (cls, p) in enumerate(new_insts):
            new_insts[i] = (cls, p * s)
        w, h = nw, nh
    return out_img, new_insts, w, h


def draw_overlay(img, insts, w=None, h=None):
    vis = img.copy()
    for _, poly in insts:
        pts = poly.copy()
        if w is not None:  # poly is normalised
            pts[:, 0] *= w
            pts[:, 1] *= h
        cv2.polylines(vis, [pts.astype(np.int32)], True, (0, 0, 255), 3)
    return vis


def main() -> None:
    args = parse_args()
    root = Path(args.data_root)
    img_dir = root / "images" / "train"
    lbl_dir = root / "labels" / "train"
    if not img_dir.exists():
        raise FileNotFoundError(f"train images not found: {img_dir}")
    print(f"[safety] augmenting TRAIN only — val/ and test/ are never touched.")

    if args.clean:
        removed = 0
        for d in (img_dir, lbl_dir):
            for f in d.glob("*_aug*"):
                f.unlink(); removed += 1
        print(f"[clean] removed {removed} existing *_aug* files.")

    srcs = sorted(p for p in img_dir.iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"} and "_aug" not in p.stem)
    if args.limit:
        srcs = srcs[: args.limit]

    dbg_dir = root / "_aug_debug"
    if args.debug_overlay:
        dbg_dir.mkdir(exist_ok=True)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    n_img = n_inst = 0
    for idx, img_path in enumerate(srcs):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[warn] unreadable, skipped: {img_path.name}")
            continue
        insts = read_label(lbl_dir / f"{img_path.stem}.txt")
        for k in range(args.n_aug):
            out_img, new_insts, w, h = augment_one(img, insts, rng, args)
            stem = f"{img_path.stem}_aug{k}"
            cv2.imwrite(str(img_dir / f"{stem}.jpg"), out_img,
                        [cv2.IMWRITE_JPEG_QUALITY, args.jpg_quality])
            write_label(lbl_dir / f"{stem}.txt", new_insts, w, h)
            n_img += 1
            n_inst += len(new_insts)
            if args.debug_overlay and idx < args.debug_overlay and k == 0:
                orig_vis = draw_overlay(img, [(c, p * np.array([img.shape[1], img.shape[0]])) for c, p in insts])
                aug_vis = draw_overlay(out_img, new_insts)
                pad = np.full((max(orig_vis.shape[0], aug_vis.shape[0]), 20, 3), 255, np.uint8)
                oh = cv2.resize(orig_vis, (int(orig_vis.shape[1] * aug_vis.shape[0] / orig_vis.shape[0]), aug_vis.shape[0]))
                combo = np.hstack([oh, pad, aug_vis])
                cv2.imwrite(str(dbg_dir / f"{img_path.stem}_overlay.jpg"), combo,
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{len(srcs)} source images -> {n_img} augmented so far")

    print(f"[done] {len(srcs)} train sources x{args.n_aug} -> {n_img} new images, {n_inst} instances "
          f"written into {img_dir}")
    if args.debug_overlay:
        print(f"[debug] overlays in {dbg_dir}")


if __name__ == "__main__":
    main()
