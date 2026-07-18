# Mask R-CNN（Torchvision）实例分割部署与数据转换指南

## 1. 基线定位

本基线使用 PyTorch 官方 `torchvision.models.detection.maskrcnn_resnet50_fpn`，骨干网络为
ResNet-50，Neck 为 FPN，检测分支输出幼果框和类别，Mask 分支为每个检测实例输出独立二值掩膜。
它是两阶段实例分割经典基线，适合与 YOLO11n-seg、RTMDet-Ins、SOLOv2 对比。

代码不依赖 Detectron2、MMDetection 或 MMCV。默认加载 COCO 预训练权重，再把分类头和掩膜头改为：

```text
类别 0：background（Torchvision 内部保留）
类别 1：orange_immature（柑橘幼果）
```

```mermaid
---
accTitle: Mask R-CNN 数据与实验流程
accDescr: 将 YOLO 多边形标签转换为 COCO，校验并可视化后训练 Torchvision Mask R-CNN，最后独立评估测试集。
---
flowchart LR
    A[YOLO polygon] --> B[prepare_dataset.py]
    B --> C[COCO JSON and images]
    C --> D[validate and visualize]
    D --> E[Mask R-CNN training]
    E --> F[model_best.pth]
    F --> G[final test evaluation]
    G --> H[COCO Mask AP]
```

## 2. 原始数据格式

当前源数据根目录为 `E:\mastercode\data\test`：

```text
data/test/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

图像和标签必须同名，例如 `images/train/abc.jpg` 对应 `labels/train/abc.txt`。每个实例占一行：

```text
class_id x1 y1 x2 y2 ... xn yn
0 0.412 0.305 0.438 0.292 0.461 0.318 ...
```

坐标是相对图像宽高归一化后的数值，范围为 `[0,1]`，每个多边形至少需要 3 个点。当前数据统计为：

| 划分 | 图像 | 实例 |
|---|---:|---:|
| train | 643 | 2,321 |
| val | 182 | 826 |
| test | 116 | 1,429 |
| 合计 | 941 | 4,576 |

> 当前划分只适合程序联调。正式论文必须按采集视频或连拍序列进行 group-aware 划分，避免相邻帧同时进入训练集和测试集。

## 3. 转换为 COCO

在 Windows PowerShell 中执行：

```powershell
cd E:\mastercode\4_baseline_choice
python scripts\prepare_dataset.py `
  --source E:\mastercode\data\test `
  --output datasets\citrus_prepared `
  --class-name orange_immature `
  --mode auto
```

`--mode auto` 优先创建硬链接，不重复占用图像空间；跨磁盘失败时自动复制。服务器上可使用同样参数：

```bash
cd /path/to/4_baseline_choice
python scripts/prepare_dataset.py \
  --source /data/citrus_yolo \
  --output datasets/citrus_prepared \
  --class-name orange_immature \
  --mode auto
```

Mask R-CNN 实际使用以下文件：

```text
datasets/citrus_prepared/coco/
├── annotations/
│   ├── instances_train.json
│   ├── instances_val.json
│   └── instances_test.json
└── images/
    ├── train/
    ├── val/
    └── test/
```

转换器把 YOLO 的类别 `0` 改成 COCO 的 `category_id=1`，同时计算像素坐标、多边形面积和
`[x, y, width, height]` 边界框。典型 COCO 标注如下：

```json
{
  "id": 1,
  "image_id": 1,
  "category_id": 1,
  "segmentation": [[125.2, 83.1, 141.8, 79.4, 153.6, 96.2]],
  "bbox": [125.2, 79.4, 28.4, 16.8],
  "area": 312.5,
  "iscrowd": 0
}
```

## 4. 转换后检查

先做结构和类别检查：

```powershell
python scripts\validate_torchvision_maskrcnn_dataset.py `
  --dataset datasets\citrus_prepared
```

输出总数应仍为 941 张图像、4,576 个实例。随后随机绘制 30 张标签图：

```powershell
python scripts\visualize_coco_dataset.py `
  --dataset datasets\citrus_prepared `
  --split train `
  --output dataset_checks\train `
  --limit 30
```

必须人工检查：每个幼果是否单独成实例、轮廓是否闭合、严重遮挡处是否误连、叶片是否被包含、相邻幼果是否合并、
小果是否漏标。标签错误应在源 YOLO 标签中修复，然后重新转换，不要直接手改 COCO JSON。

## 5. 配置 PyTorch 环境

建议使用 Python 3.10，并给 Mask R-CNN 建立独立环境：

```bash
conda create -n citrus_maskrcnn python=3.10 -y
conda activate citrus_maskrcnn
```

先在 PyTorch 官方安装选择器中根据服务器驱动选择 CUDA wheel。下面仅以 CUDA 12.8 为例：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
cd /path/to/4_baseline_choice
pip install -r requirements-torchvision-maskrcnn.txt
```

不要把 CPU 版 `torch` 与 CUDA 版 `torchvision` 混装。当前本机环境是
`torch 2.8.0+cpu / torchvision 0.23.0+cpu`，只能做代码检查，不能进行可用速度的训练。服务器安装后运行：

```bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -c "from torchvision.ops import nms; import torch; print(nms(torch.tensor([[0.,0.,1.,1.]]), torch.tensor([1.]), 0.5))"
python -c "import pycocotools; print('pycocotools OK')"
```

第二条命令用于提前发现 `torchvision::nms` 编译版本不匹配。

## 6. 训练

先运行 2 epoch 冒烟实验，确认显存、损失、验证和保存流程全部正常：

```bash
python scripts/train_torchvision_maskrcnn.py \
  --dataset datasets/citrus_prepared \
  --name MRCNN_R50_FPN_smoke \
  --epochs 2 \
  --batch 2 \
  --workers 2 \
  --val-interval 1 \
  --save-interval 1
```

正式单卡实验：

```bash
python scripts/train_torchvision_maskrcnn.py \
  --dataset datasets/citrus_prepared \
  --name E_maskrcnn_r50_fpn_seed42 \
  --epochs 300 \
  --batch 2 \
  --workers 8 \
  --lr 0.005 \
  --imgsz 640 \
  --seed 42 \
  --val-interval 5 \
  --save-interval 25
```

默认设置为 COCO 预训练、SGD、momentum `0.9`、weight decay `0.0005`、水平翻转和 CUDA AMP。
输入保持宽高比，长边缩放到不超过 640，模型内部再完成 batch padding，不会把果实强制拉伸为正方形。
首次运行会下载官方 COCO 权重到 PyTorch 缓存目录。

显存不足时使用 `--batch 1 --lr 0.0025`；不要只减 batch 而保持原学习率。关闭混合精度可加
`--no-amp`。中断后续训：

```bash
python scripts/train_torchvision_maskrcnn.py \
  --dataset datasets/citrus_prepared \
  --name E_maskrcnn_r50_fpn_seed42 \
  --epochs 300 \
  --resume runs/torchvision_maskrcnn/E_maskrcnn_r50_fpn_seed42/model_last.pth
```

重要产物：

```text
runs/torchvision_maskrcnn/E_maskrcnn_r50_fpn_seed42/
├── model_best.pth
├── model_last.pth
├── run_metadata.json
├── history.json
├── validation/best_metrics.json
└── checkpoints/epoch_*.pth
```

## 7. 最终测试

模型和超参数根据验证集确定后，只使用最佳权重评估一次测试集：

```bash
python scripts/eval_torchvision_maskrcnn.py \
  --weights runs/torchvision_maskrcnn/E_maskrcnn_r50_fpn_seed42/model_best.pth \
  --dataset datasets/citrus_prepared \
  --split test \
  --output runs/evaluation/E_maskrcnn_r50_fpn_seed42_test
```

`metrics.json` 包含 Mask AP50-95、AP50、AP75、APsmall/APmedium/APlarge、固定阈值下的
precision/recall/F1、参数量和单张模型推理时间；`predictions.coco.json` 可用于统一复核。论文主表至少报告：

```text
Mask AP50-95, Mask AP50, Precision, Recall, Params, GFLOPs, latency, peak VRAM
```

筛选实验可只跑 seed 42；最终 baseline 和最终方法应使用 `42、3407、2026` 三个种子，并报告均值和标准差。

## 8. 服务器迁移

需要上传：

```text
4_baseline_choice/scripts/
4_baseline_choice/configs/
4_baseline_choice/requirements-torchvision-maskrcnn.txt
4_baseline_choice/datasets/citrus_prepared/
```

不需要上传 `detectron2-main` 或 MMDetection。COCO JSON 内只记录图像文件名，因此移动整个
`citrus_prepared` 目录后无需修改绝对路径。建议同时记录服务器 GPU、驱动、PyTorch 版本、Git revision、
完整训练命令和随机种子。

## 9. 常见故障

| 现象 | 原因与处理 |
|---|---|
| `torch.cuda.is_available() == False` | 安装了 CPU wheel，或 NVIDIA 驱动不可见；重新按官方选择器安装 CUDA wheel。 |
| `operator torchvision::nms does not exist` | `torch` 与 `torchvision` 版本或 CUDA 后缀不匹配；同时卸载后成对重装。 |
| `No module named pycocotools` | 执行 `pip install pycocotools`。 |
| AP 全为 0 | 检查 COCO 类别必须为 `1`，模型前景类别也为 `1`；先运行验证脚本和可视化脚本。 |
| loss 为 NaN | 常见于退化多边形、零面积框或过大学习率；重新验证数据并检查最近修改的标签。 |
| CUDA OOM | 改为 `--batch 1 --lr 0.0025`，再考虑减小 `--imgsz`；正式对比中所有模型必须使用同一输入协议。 |
| 首次运行无法下载权重 | 在联网机器先运行一次，随后把 PyTorch 缓存中的 Mask R-CNN 权重文件复制到服务器同一缓存目录。 |

## 10. 官方依据

1. PyTorch Torchvision Mask R-CNN R50-FPN model documentation.
2. Torchvision Object Detection Finetuning Tutorial.
3. Torchvision detection model source and COCO pretrained-weight definitions.

E:\mastercode\data\orange_yolo