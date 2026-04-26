import argparse
import os

import torch
import torch.nn as nn

from train import MiniSegNet, build_dataloader, normalize_mask


@torch.no_grad()
def evaluate(args):
    '''
    执行最小化评估流程。

    Args:
        args (argparse.Namespace): 评估参数。

    Returns:
        None

    Notes:
        会从 checkpoint 中加载模型参数, 并输出平均 loss 与 IoU。
    '''
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    dataloader = build_dataloader(args)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    model = MiniSegNet().to(device)
    model.load_state_dict(model_state, strict=True)
    model.eval()

    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_iou = 0.0
    total_batches = 0

    for image, mask in dataloader:
        image = image.to(device).float()
        mask = (normalize_mask(mask) > 0).float().to(device)

        logits = model(image)
        loss = criterion(logits, mask)

        prob = torch.sigmoid(logits)
        pred = (prob >= args.threshold).float()

        intersection = (pred * mask).sum(dim=(1, 2, 3))
        union = ((pred + mask) > 0).float().sum(dim=(1, 2, 3)).clamp_min(1.0)
        batch_iou = (intersection / union).mean().item()

        total_loss += loss.item()
        total_iou += batch_iou
        total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)
    avg_iou = total_iou / max(total_batches, 1)

    print(f"Evaluate done | loss={avg_loss:.6f} | iou={avg_iou:.6f} | batches={total_batches}")


def parse_args():
    '''
    解析评估命令行参数。

    Returns:
        argparse.Namespace: 解析后的参数对象。
    '''
    parser = argparse.ArgumentParser(description="Minimal evaluation entry for 0_segment.")
    parser.add_argument("--checkpoint", type=str, default=os.path.join("checkpoints", "minimal_last.pt"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--image-dir", type=str, default="")
    parser.add_argument("--label-dir", type=str, default="")
    parser.add_argument("--label-type", type=str, default="mask", choices=["mask", "txt", "json", "npy"])

    parser.add_argument("--synthetic-length", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint-name", type=str, default="minimal_last.pt")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
