"""
FCN 推理 + 可视化脚本
用法：
    # 单张图片推理
    python inference.py --img path/to/image.jpg --ckpt checkpoints/fcn8s_best.pth \
                        --model fcn8s --num_classes 21

    # 批量推理整个文件夹
    python inference.py --img_dir path/to/images/ --ckpt checkpoints/fcn8s_best.pth \
                        --model fcn8s --num_classes 21 --out_dir results/
"""

import os
import argparse
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision import transforms

from model import build_model

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def load_model(ckpt_path, model_type, num_classes, device):
    model = build_model(model_type, num_classes)
    ckpt  = torch.load(ckpt_path, map_location=device)

    # 兼容保存格式
    state = ckpt.get('model_state', ckpt)
    model.load_state_dict(state)
    model.to(device).eval()

    epoch = ckpt.get('epoch', '?')
    miou  = ckpt.get('miou',  '?')
    print(f"模型加载成功：epoch={epoch}, mIoU={miou}")
    return model


def preprocess(img_path, img_size):
    img = Image.open(img_path).convert('RGB')
    orig_size = img.size  # (W, H)
    img_resized = img.resize((img_size, img_size), Image.BILINEAR)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    tensor = transform(img_resized).unsqueeze(0)  # (1, 3, H, W)
    return tensor, img, orig_size


def generate_colormap(num_classes):
    """生成不同颜色用于可视化各类别"""
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # 背景为黑色
    return colors


def predict_single(model, img_path, img_size, num_classes, device, out_path=None,
                   class_names=None, show=True):
    tensor, orig_img, orig_size = preprocess(img_path, img_size)
    tensor = tensor.to(device)

    with torch.no_grad():
        output = model(tensor)              # (1, C, H, W)
        pred   = output.argmax(dim=1)[0]    # (H, W)
        pred   = pred.cpu().numpy()

    # 生成彩色分割图
    colormap = generate_colormap(num_classes)
    seg_color = colormap[pred]              # (H, W, 3)
    seg_img = Image.fromarray(seg_color).resize(orig_size, Image.NEAREST)

    # 统计出现的类别
    present_classes = np.unique(pred)

    # ── 可视化 ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(orig_img)
    axes[0].set_title('原始图像')
    axes[0].axis('off')

    axes[1].imshow(seg_img)
    axes[1].set_title('分割结果')
    axes[1].axis('off')

    # 叠加显示
    blend = Image.blend(orig_img.convert('RGBA'),
                        seg_img.convert('RGBA'), alpha=0.5)
    axes[2].imshow(blend)
    axes[2].set_title('叠加效果')
    axes[2].axis('off')

    # 图例
    if class_names:
        patches = []
        for cls_id in present_classes:
            if cls_id < len(class_names):
                name  = class_names[cls_id]
                color = colormap[cls_id] / 255.0
                patches.append(mpatches.Patch(color=color, label=f'{cls_id}: {name}'))
        if patches:
            fig.legend(handles=patches, loc='lower center',
                       ncol=min(len(patches), 6), fontsize=8)

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"结果已保存：{out_path}")
    if show:
        plt.show()
    plt.close()

    return pred


def predict_folder(model, img_dir, img_size, num_classes, device, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    files = [f for f in os.listdir(img_dir) if f.lower().endswith(exts)]

    print(f"共 {len(files)} 张图像，开始推理...")

    for fname in files:
        img_path = os.path.join(img_dir, fname)
        out_path = os.path.join(out_dir, fname.rsplit('.', 1)[0] + '_seg.png')
        predict_single(model, img_path, img_size, num_classes, device,
                       out_path=out_path, show=False)

    print(f"全部完成，结果保存在：{out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FCN 推理脚本')

    parser.add_argument('--ckpt',        type=str, required=True,  help='模型checkpoint路径')
    parser.add_argument('--model',       type=str, default='fcn8s',
                        choices=['fcn32s', 'fcn16s', 'fcn8s'])
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--img_size',    type=int, default=512)

    # 单张 or 批量
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--img',     type=str, help='单张图片路径')
    group.add_argument('--img_dir', type=str, help='图片文件夹路径（批量）')

    parser.add_argument('--out_dir', type=str, default='results', help='批量推理输出目录')
    parser.add_argument('--out',     type=str, default=None,      help='单张推理保存路径')
    parser.add_argument('--no_show', action='store_true',         help='不弹窗显示')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = load_model(args.ckpt, args.model, args.num_classes, device)

    if args.img:
        predict_single(model, args.img, args.img_size, args.num_classes,
                       device, out_path=args.out, show=not args.no_show)
    else:
        predict_folder(model, args.img_dir, args.img_size, args.num_classes,
                       device, out_dir=args.out_dir)
