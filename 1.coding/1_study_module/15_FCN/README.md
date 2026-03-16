# FCN 复现项目

PyTorch 实现 FCN-32s / FCN-16s / FCN-8s，支持自定义数据集。

## 文件结构

```
fcn/
├── model.py       # FCN-32s / 16s / 8s 模型定义
├── dataset.py     # 数据集加载器
├── train.py       # 训练 + 验证脚本
├── inference.py   # 推理 + 可视化脚本
└── README.md
```

## 安装依赖

```bash
pip install torch torchvision numpy pillow matplotlib
```

## 数据集目录结构

```
your_dataset/
├── images/
│   ├── train/    ← 训练图像 (*.jpg 或 *.png)
│   └── val/      ← 验证图像
└── masks/
    ├── train/    ← 训练掩码 (*.png，像素值 = 类别ID)
    └── val/      ← 验证掩码
```

> 掩码文件名必须与图像文件名一致（后缀可不同）。  
> 255 像素值表示"忽略区域"，训练时不计入损失。

## 训练

### 训练 FCN-8s（推荐）
```bash
python train.py \
    --data_root /path/to/your_dataset \
    --num_classes 21 \          # 改成你的类别数（含背景）
    --model fcn8s \
    --epochs 100 \
    --batch_size 4 \
    --lr 1e-3 \
    --img_size 512 \
    --save_dir checkpoints/
```

### 依次训练三个版本（论文建议的初始化方式）
```bash
# 第一步：训练 FCN-32s
python train.py --model fcn32s --num_classes 21 --data_root /path/to/data ...

# 第二步：训练 FCN-16s（用 FCN-32s 权重初始化更快收敛）
python train.py --model fcn16s --num_classes 21 --data_root /path/to/data ...

# 第三步：训练 FCN-8s（同上）
python train.py --model fcn8s  --num_classes 21 --data_root /path/to/data ...
```

## 推理

### 单张图片
```bash
python inference.py \
    --ckpt checkpoints/fcn8s_best.pth \
    --model fcn8s \
    --num_classes 21 \
    --img /path/to/image.jpg \
    --out result.png
```

### 批量推理整个文件夹
```bash
python inference.py \
    --ckpt checkpoints/fcn8s_best.pth \
    --model fcn8s \
    --num_classes 21 \
    --img_dir /path/to/images/ \
    --out_dir results/
```

## 关键超参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lr` | 1e-3 | 学习率，论文使用固定学习率 |
| `--batch_size` | 4 | 显存不够可调为 2 |
| `--img_size` | 512 | 输入尺寸，建议512或更大 |
| `--lr_step` | 50 | 学习率衰减间隔 |
| `--num_classes` | - | **必须填**，含背景类 |

## 常见问题

**Q: CUDA out of memory**  
A: 减小 `--batch_size`（调为2或1），或减小 `--img_size`（调为256或320）。

**Q: mIoU 很低**  
A: 先检查掩码的像素值范围是否正确（应该是 0 ~ num_classes-1）。  
可以用以下代码验证：
```python
from PIL import Image
import numpy as np
mask = Image.open('your_mask.png')
print(np.unique(np.array(mask)))  # 应该输出类别ID，如 [0, 1, 2, ...]
```

**Q: 文件名匹配失败**  
A: 确保图像和掩码文件名相同（后缀不同没关系），如 `cat.jpg` 对应 `cat.png`。
