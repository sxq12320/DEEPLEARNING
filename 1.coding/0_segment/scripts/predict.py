"""分割模型预测 CLI 入口。"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import MiniSegNet


def parse_args():
    """解析预测命令行参数。

    Returns:
        argparse.Namespace: 解析后的参数对象。
    """
    parser = argparse.ArgumentParser(description="Segmentation model prediction")
    parser.add_argument("--source", type=str, required=True, help="Input image path")
    parser.add_argument("--weights", type=str, default="runs/train/exp/weights/best.pt")
    parser.add_argument("--imgsz", type=int, default=128, help="Model input size")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--project", type=str, default="runs/predict")
    parser.add_argument("--name", type=str, default="exp")
    parser.add_argument("--show", action="store_true", help="Display the result")
    return parser.parse_args()


def predict(args):
    """执行单张图像分割预测并保存可视化结果。

    Args:
        args (argparse.Namespace): 预测配置参数。

    Returns:
        None: 本函数无显式返回值。
    """
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    model = MiniSegNet().to(device)
    if args.weights and os.path.isfile(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"Loaded weights from: {args.weights}")
    else:
        print("No weight file provided, using random initialization.")

    model.eval()

    if not os.path.isfile(args.source):
        raise FileNotFoundError(f"Source image not found: {args.source}")

    img_bgr = cv2.imread(args.source)
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {args.source}")

    orig_h, orig_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        prob = torch.sigmoid(logits)
        mask_pred = (prob > 0.5).float().cpu().numpy()[0, 0]

    mask_full = cv2.resize(mask_pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    overlay = img_bgr.copy()
    mask_bool = mask_full > 0.5
    overlay[mask_bool] = (overlay[mask_bool] * 0.6 + np.array([0, 255, 0]) * 0.4).astype(np.uint8)

    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{Path(args.source).stem}_seg.jpg"
    cv2.imwrite(str(save_path), overlay)
    print(f"Prediction saved to: {save_path}")

    if args.show:
        cv2.imshow("Segmentation", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    predict(parse_args())
