from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


DATA_ROOT = Path(r"E:\mastercode\data\test")
MODEL_PATH = Path(
    r"E:\mastercode\ultralytics-main-new\1_results\ORANGE_WUXI_SEG"
    r"\005_yolo11n_seg_orange_wuxi_597imgs_baseline\weights\best.pt"
)
OUT_ROOT = Path(
    r"E:\mastercode\ultralytics-main-new\1_results\ORANGE_WUXI_SEG"
    r"\005_yolo11n_seg_orange_wuxi_597imgs_baseline\mask_overlay_compare"
)

SPLITS = ("val", "test")
CONF = 0.25
IMG_SIZE = 640


def draw_header(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(out, text, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def overlay_polygons(image: np.ndarray, polygons: list[np.ndarray], color: tuple[int, int, int]) -> np.ndarray:
    out = image.copy()
    mask_layer = np.zeros_like(image)
    for poly in polygons:
        if len(poly) >= 3:
            cv2.fillPoly(mask_layer, [poly.astype(np.int32)], color)
            cv2.polylines(out, [poly.astype(np.int32)], True, color, 2, cv2.LINE_AA)
    return cv2.addWeighted(out, 0.72, mask_layer, 0.28, 0)


def load_gt_polygons(label_path: Path, width: int, height: int) -> list[np.ndarray]:
    polygons: list[np.ndarray] = []
    if not label_path.exists():
        return polygons
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        coords = np.array([float(v) for v in parts[1:]], dtype=np.float32).reshape(-1, 2)
        coords[:, 0] *= width
        coords[:, 1] *= height
        polygons.append(coords)
    return polygons


def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] == height:
        return image
    scale = height / image.shape[0]
    width = int(round(image.shape[1] * scale))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def main() -> None:
    model = YOLO(str(MODEL_PATH))
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        image_dir = DATA_ROOT / "images" / split
        label_dir = DATA_ROOT / "labels" / split
        out_dir = OUT_ROOT / split
        out_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(
            [p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )
        for image_path in image_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                print(f"skip unreadable image: {image_path}")
                continue

            height, width = image.shape[:2]
            gt_polys = load_gt_polygons(label_dir / f"{image_path.stem}.txt", width, height)
            gt_view = overlay_polygons(image, gt_polys, (0, 220, 0))

            result = model.predict(
                source=str(image_path),
                imgsz=IMG_SIZE,
                conf=CONF,
                device="cpu",
                retina_masks=True,
                verbose=False,
            )[0]
            pred_polys = []
            if result.masks is not None:
                pred_polys = [np.asarray(poly, dtype=np.float32) for poly in result.masks.xy]
            pred_view = overlay_polygons(image, pred_polys, (0, 0, 255))

            original = draw_header(image, "Original")
            gt_view = draw_header(gt_view, f"GT masks: {len(gt_polys)}")
            pred_view = draw_header(pred_view, f"Pred masks: {len(pred_polys)}  conf>={CONF}")

            target_h = min(900, height)
            panels = [resize_to_height(panel, target_h) for panel in (original, gt_view, pred_view)]
            compare = cv2.hconcat(panels)
            cv2.imwrite(str(out_dir / f"{image_path.stem}_compare.jpg"), compare)

        print(f"{split}: wrote {len(image_paths)} compare images to {out_dir}")


if __name__ == "__main__":
    main()
