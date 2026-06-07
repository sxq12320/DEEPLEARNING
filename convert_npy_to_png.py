"""
将 apple_rgbd_amoal 目录下的 4 通道 float32 .npy 文件批量转换为 RGBA .png
深度通道 (ch3, 0~5.4) 会被线性缩放到 0~255 存入 A 通道。

转换后单张从 ~26MB 降至 ~2MB，总大小从 ~99GB 降至 ~7GB。

用法:
    python convert_npy_to_png.py
    python convert_npy_to_png.py --src /path/to/dataset --dst /path/to/output
    python convert_npy_to_png.py --delete-original   # 转换后删除原 npy
"""

import os
import glob
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def convert_one(src_path: str, dst_path: str):
    """单张转换: npy (H,W,4) float32 -> RGBA PNG uint8 (每张独立归一化深度)"""
    arr = np.load(src_path)  # (H, W, 4), float32

    rgb = arr[:, :, :3].astype(np.uint8)  # 0~255 直接截断
    depth = arr[:, :, 3]

    # 每张图独立归一化深度到 0~255
    d_max = depth.max()
    if d_max == 0:
        d_max = 1.0

    depth_u8 = np.clip(depth / d_max * 255, 0, 255).astype(np.uint8)

    rgba = np.concatenate([rgb, depth_u8[:, :, np.newaxis]], axis=-1)
    img = Image.fromarray(rgba, "RGBA")
    img.save(dst_path, "PNG", optimize=True)


def main():
    parser = argparse.ArgumentParser(description="Convert 4-ch float32 npy to RGBA PNG")
    parser.add_argument("--src", default=r"E:\mastercode\data\Apple_RGB_D_Amoal\apple_rgbd_amoal",
                        help="源数据集根目录 (含 train/val/test 子目录)")
    parser.add_argument("--dst", default=None,
                        help="输出目录，默认在 src 同级创建 _png 后缀目录")
    parser.add_argument("--delete-original", action="store_true",
                        help="转换成功后删除原始 npy 文件")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="跳过已存在的 PNG 文件")
    args = parser.parse_args()

    src_root = Path(args.src)
    if args.dst:
        dst_root = Path(args.dst)
    else:
        dst_root = src_root.parent / (src_root.name + "_png")

    # 收集所有 npy 文件
    npy_files = sorted(glob.glob(str(src_root / "**" / "*.npy"), recursive=True))
    print(f"找到 {len(npy_files)} 个 npy 文件")
    print(f"源目录: {src_root}")
    print(f"目标目录: {dst_root}")

    if not npy_files:
        print("未找到 npy 文件，请检查路径")
        return

    # 转换 (每张独立归一化深度)
    errors = []
    skipped = 0
    for src_path in tqdm(npy_files, desc="转换"):
        rel = os.path.relpath(src_path, src_root)
        dst_path = os.path.join(dst_root, rel).replace(".npy", ".png")
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        # 跳过已存在且非空的 PNG
        if args.skip_existing and os.path.exists(dst_path) and os.path.getsize(dst_path) > 100:
            skipped += 1
            continue

        try:
            convert_one(src_path, dst_path)
            if args.delete_original:
                os.remove(src_path)
        except Exception as e:
            errors.append((src_path, str(e)))
            print(f"\n错误: {src_path} -> {e}")

    print(f"\n完成! 成功: {len(npy_files) - len(errors) - skipped}, 跳过: {skipped}, 失败: {len(errors)}")
    if errors:
        print("失败文件:")
        for f, e in errors:
            print(f"  {f}: {e}")


if __name__ == "__main__":
    main()
