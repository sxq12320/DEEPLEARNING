# 服务器代码包使用说明

这个目录是从本地代码同步出来的服务器代码包，不包含训练结果和数据集。

## 目录

- `baseline_choice/`：YOLO、Mask R-CNN、RF-DETR、MMDetection、U-Net 等 baseline 训练与数据转换脚本。
- `ultralytics-main-new/`：本地 Ultralytics 改造代码，包括橙子分割 YAML、StarNet 模块和训练评估脚本。
- `yolo26n.pt`：YOLO26n 权重文件。

## 数据路径

复制到服务器后，推荐把数据放到服务器固定目录，例如：

```bash
/data/citrus/orange_wuxi
/data/citrus/orange_yolo
/data/citrus/runs
```

如果要从 LabelMe 数据重新生成 YOLO 和所有 baseline 格式，执行：

```bash
cd /path/to/code/baseline_choice
export ORANGE_LABELME_DIR=/data/citrus/orange_wuxi
export ORANGE_YOLO_DIR=/data/citrus/orange_yolo
python run_update_orange_dataset.py
```

该命令会生成：

```bash
$ORANGE_YOLO_DIR
/path/to/code/baseline_choice/datasets/citrus_prepared
```

当前默认策略是不增强，只将全部原图随机打乱后按 `train:val:test = 7:2:1` 划分。

## 训练入口

进入 `baseline_choice/` 后，按需要运行：

```bash
python run_yolo_baselines.py
python run_maskrcnn.py
python run_unet.py
python run_rfdetr.py
python run_mmdet.py
```

这些脚本里有 `SERVER_*` 路径配置。如果服务器数据不在 `/data/citrus/`，修改对应 `SERVER_SOURCE_DATASET`、`SERVER_PREPARED_DATASET`、`SERVER_OUTPUT_ROOT` 即可。

Ultralytics 自定义模型在：

```bash
cd /path/to/code/ultralytics-main-new
pip install -e .
python train_citrus_seg.py --model 0_orange_yaml/001_yolo11-seg.yaml --name E0_yolo11n
```
