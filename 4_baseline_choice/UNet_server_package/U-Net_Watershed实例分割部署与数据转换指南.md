# U-Net + Watershed 柑橘幼果实例分割部署与数据转换指南

## 1. 基线定位

本基线使用 `segmentation_models_pytorch.Unet`，编码器为 ResNet18。U-Net 原生输出的是二值语义掩膜，不能区分相互接触的多个幼果，因此评估流程为：

```text
RGB 图像
  -> U-Net 前景概率图
  -> 阈值二值化
  -> 距离变换
  -> 局部极大值生成种子
  -> Watershed 分离粘连区域
  -> 恢复到原图尺寸
  -> COCO 实例分割评估
```

论文表格中的主要指标必须使用 `mask_ap_50_95`，不能只用 Dice 或 IoU 与 Mask R-CNN、YOLO-seg 直接比较。Dice/IoU 仅反映整体前景分割质量，Watershed 后的 COCO Mask AP 才反映实例级效果。

## 2. 一键运行文件

入口是：

```text
4_baseline_choice/run_unet.py
```

只修改该文件顶部 `USER SETTINGS` 区域，不需要手写长命令，也不需要 BAT 或 SH 文件。直接在 PyCharm、VS Code 中运行该 Python 文件即可。

推荐顺序：

1. `RUN_MODE = "smoke"`：2 个 epoch，小 batch，检查完整流程。
2. `RUN_MODE = "train"`：正式训练，自动保存最优权重和断点。
3. `RUN_MODE = "test"`：使用正式实验最优权重评估 test。
4. `RUN_MODE = "all"`：正式训练结束后自动测试。
5. `RUN_MODE = "prepare"`：只转换并检查数据。

程序启动后会打印 Python、Torch、CUDA、GPU、数据路径、模型、编码器、输入尺寸、batch、epoch、优化器、学习率、损失、阈值和全部 Watershed 参数。

运行过程中会显示以下进度条：

```text
Convert train:       659/659 images
Train epoch 1:       330/330 batches  loss=... dice=... lr=...
Validate epoch 1:     47/47 batches   dice=... iou=... instances=...
Test test:            24/24 batches   dice=... iou=... instances=...
```

## 3. 原始数据格式

当前原始数据集：

```text
E:\mastercode\data\orange_yolo
├── train
│   ├── images
│   └── labels
├── val
│   ├── images
│   └── labels
├── test
│   ├── images
│   └── labels
└── data.yaml
```

转换器也支持 `images/train + labels/train` 布局。每张图片必须有同名 `.txt`；无目标图片保留空标签文件。

每行标签必须是 YOLO 实例分割多边形：

```text
class_id x1 y1 x2 y2 x3 y3 ... xn yn
```

坐标均归一化到 `[0, 1]`，至少 3 个点。例如：

```text
0 0.312 0.224 0.401 0.218 0.438 0.351 0.327 0.372
```

不能把检测框格式 `class x_center y_center width height` 当成分割标签。

## 4. 自动转换结果

`run_unet.py` 首次运行时会调用 `scripts/prepare_dataset.py`，生成：

```text
datasets/citrus_prepared
├── semantic
│   ├── images/{train,val,test}
│   └── masks/{train,val,test}/*.png
└── coco
    ├── images/{train,val,test}
    └── annotations/instances_{train,val,test}.json
```

`semantic/masks` 是 U-Net 的二值训练标签，背景像素为 `0`，全部幼果前景为 `255`。COCO JSON 保留每个幼果的独立多边形，用于 Watershed 结果的实例级评估。

当前数据统计：

| 划分 | 图像 | 实例 | 负样本图像 |
|---|---:|---:|---:|
| train | 659 | 3,483 | 10 |
| val | 188 | 672 | 3 |
| test | 94 | 421 | 1 |

正式论文实验前仍应按连续采集序列进行 group-aware 重划分，避免同一 burst 的近重复帧跨 train/val/test。

## 5. 本机环境配置

先激活已有的 CUDA PyTorch 环境。不要让 `pip` 自动替换已经匹配好的 Torch 和 Torchvision：

```powershell
conda activate yolo
cd E:\mastercode\4_baseline_choice
python -m pip install -r requirements-unet.txt
```

当前已验证组合：

```text
Python 环境：E:\AppInstallion\0_4_annaconda\envs\yolo
PyTorch：2.5.0+cu118
Torchvision：0.20.0+cu118
segmentation-models-pytorch：0.5.0
timm：1.0.28
```

首次使用 `ENCODER_WEIGHTS = "imagenet"` 时需要下载 ResNet18 预训练权重。服务器不能联网时，可先在联网机器完成一次模型构建并复制缓存，或临时改成：

```python
ENCODER_WEIGHTS = "none"
```

后者属于不同初始化协议，正式对比时所有随机种子必须保持相同初始化方式。

## 6. USER SETTINGS 参数

### 路径

Windows 自动使用：

```python
WINDOWS_SOURCE_DATASET = Path(r"E:\mastercode\data\orange_yolo")
WINDOWS_PREPARED_DATASET = Path(
    r"E:\mastercode\4_baseline_choice\datasets\citrus_prepared"
)
```

服务器自动使用 `SERVER_*` 路径。只需按服务器实际目录修改：

```python
SERVER_SOURCE_DATASET = Path("/data/citrus/orange_yolo")
SERVER_PREPARED_DATASET = Path("/data/citrus/citrus_prepared")
SERVER_OUTPUT_ROOT = Path("/data/citrus/runs/unet_watershed")
SERVER_EVALUATION_ROOT = Path("/data/citrus/runs/evaluation")
```

### 训练参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `FORMAL_EPOCHS` | 300 | 正式训练轮数 |
| `ENCODER` | resnet18 | U-Net 编码器 |
| `BATCH_SIZE` | 8 | 显存不足时降为 4 或 2 |
| `LEARNING_RATE` | 0.0003 | AdamW 初始学习率 |
| `WEIGHT_DECAY` | 0.0001 | 权重衰减 |
| `IMAGE_SIZE` | 640 | 保持比例后 letterbox 到 640×640 |
| `WORKERS` | 8 | DataLoader 进程数 |
| `USE_AMP` | True | CUDA 混合精度 |
| `SEED` | 42 | 随机种子 |

损失固定为 `BCEWithLogits + DiceLoss`，优化器为 AdamW，调度器为 CosineAnnealingLR。训练时仅做水平翻转，以保持基线清晰。

### Watershed 参数

| 参数 | 默认值 | 调大后的主要效果 |
|---|---:|---|
| `PROBABILITY_THRESHOLD` | 0.50 | 前景更保守，漏检可能增加 |
| `WATERSHED_MIN_DISTANCE` | 8 | 种子更少，减少过分割但可能欠分割 |
| `WATERSHED_MIN_AREA` | 20 | 删除更多小噪声实例 |
| `MAX_INSTANCES` | 50 | 每图最多输出实例数 |

这些值作用于 640 尺度下去除 padding 后的有效区域。建议先在 val 上进行小范围搜索：

```text
probability threshold: 0.40, 0.50, 0.60
min distance:           6, 8, 10, 12
min area:               10, 20, 40
```

确定后锁定参数，只在 test 上评估一次，禁止根据 test 指标反复调参。

## 7. 输出与指标

训练结果：

```text
runs/unet_watershed/<实验名>/
├── model_best.pth
├── model_last.pth
├── history.json
├── run_metadata.json
├── checkpoints/
└── validation/
    ├── best_metrics.json
    └── best_predictions.coco.json
```

测试结果：

```text
runs/evaluation/<实验名>_test/
├── metrics.json
└── predictions.coco.json
```

控制台和 JSON 会输出：

- `semantic_dice`、`semantic_iou`：U-Net 二值语义质量。
- `mask_ap_50_95`、`mask_ap_50`、`mask_ap_75`：实例分割主指标。
- `mask_ap_small/medium/large`：不同尺度实例 AP。
- `mask_precision`、`mask_recall`、`mask_f1`：固定 IoU/置信度下的统计。
- `params_m`：模型参数量，ResNet18 U-Net 约 14.328M。
- `model_latency_ms_per_image`：仅模型前向时间，不包含 Watershed。
- `peak_vram_mb`：推理阶段峰值显存。
- `prediction_count`：输出实例总数。

`model_last.pth` 每个 epoch 更新，程序中断后再次运行同一个模式会自动续训；完成的实验不会被覆盖。

## 8. 服务器部署

将 `UNet_server_package.zip` 和原始数据集上传服务器。解压后目录应为：

```text
UNet_server_package/
├── run_unet.py
├── requirements-unet.txt
├── scripts/
└── tests/
```

服务器环境示例：

```bash
conda create -n citrus_unet python=3.10 -y
conda activate citrus_unet
# 按服务器 CUDA 版本安装匹配的 torch 和 torchvision
python -m pip install -r requirements-unet.txt
```

修改 `run_unet.py` 中四个 `SERVER_*` 路径和 `DEVICE` 后，直接运行：

```bash
python run_unet.py
```

使用 `tmux` 或 `screen` 可以避免 SSH 断线终止训练；这不改变一键入口。先以 `smoke` 检查 CUDA、数据、预训练权重和完整评估，再切换为 `train`。

## 9. 公平对比要求

U-Net、Mask R-CNN 和 YOLO-seg 必须使用完全相同的 train/val/test 图像集合与原始 COCO 实例标注。统一报告 test 的 Mask AP、参数量、输入尺寸和同一硬件上的延迟。

U-Net 是语义分割后处理基线，其 Watershed 对阈值敏感，主要用于回答“经典编码器-解码器加传统实例分离能达到什么水平”。若粘连和遮挡场景明显落后于原生实例网络，这本身就是柑橘密集幼果实例分割难点的有效实验依据。

## 10. 常见问题

### 显存不足

依次把 `BATCH_SIZE` 从 8 降到 4、2、1。不要首先修改 `IMAGE_SIZE`，否则会破坏与其他 640 输入基线的公平性。

### 找不到 segmentation_models_pytorch

确认 IDE 使用的是安装依赖的同一个 Python，并执行：

```text
python -c "import segmentation_models_pytorch; print(segmentation_models_pytorch.__version__)"
```

### 找不到数据

查看程序启动时打印的 `source` 和 `prepared`。Windows 使用 `WINDOWS_*`，Linux 自动使用 `SERVER_*`。

### 粘连幼果只输出一个实例

优先减小 `WATERSHED_MIN_DISTANCE`，再检查概率图中两个幼果中心是否形成独立峰值。若 U-Net 已将连接区域预测成近似均匀概率，仅调整后处理不能完全解决。

### 一个幼果被拆成多个实例

增大 `WATERSHED_MIN_DISTANCE` 或 `WATERSHED_MIN_AREA`。所有正式结果必须记录最终参数，不能逐图手工调整。
