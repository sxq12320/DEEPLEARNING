# Light v3 系列服务器运行说明

以下命令假设代码位于 `/data/sxq/code/ultralytics-main-new`。`--data` 必须由你填写服务器上真实的 `data.yaml`；程序只保存快照，不会猜测路径，也没有数据指纹确认步骤。

## 1. 安装和构建检查

```bash
cd /data/sxq/code/ultralytics-main-new
pip install -e .

python 20260830_citrus_light_batch.py \
  --data /data/sxq/datasets/orange_yolo/data.yaml \
  --suite all \
  --dry-run
```

应依次看到 Light00、01、02、05、06、03、04、07 共 8 条 `BUILD OK`。

## 2. 先做 3 epoch smoke

```bash
python -u 20260830_citrus_light_batch.py \
  --data /data/sxq/datasets/orange_yolo/data.yaml \
  --suite smoke \
  --epochs 3 \
  --device 0 \
  --project /data/sxq/results/Light/CITRUS_LIGHT_V3_SMOKE_3EP
```

8 个模型会在同一张卡上顺序运行，不会并行抢卡。smoke 全部完成后再进入 50 epoch。

## 3. 50 epoch 纯结构筛选（第一优先级）

```bash
nohup python -u 20260830_citrus_light_batch.py \
  --data /data/sxq/datasets/orange_yolo/data.yaml \
  --suite screen \
  --epochs 50 \
  --device 0 \
  --project /data/sxq/results/Light/CITRUS_LIGHT_V3_STRUCTURE_50EP \
  > light_v3_structure_50ep.log 2>&1 &

tail -f light_v3_structure_50ep.log
```

该队列只跑 Light00、01、02、05、06，用于分离 PConv、RepMixer 和 AFPN 的独立作用及交互。不要一开始直接跑 300 epoch：G0830 的峰值集中在 54--87 epoch，300 epoch 末值普遍回落。

## 4. 50 epoch Pareto 候选（结构筛选后再跑）

```bash
nohup python -u 20260830_citrus_light_batch.py \
  --data /data/sxq/datasets/orange_yolo/data.yaml \
  --suite pareto \
  --epochs 50 \
  --device 0 \
  --project /data/sxq/results/Light/CITRUS_LIGHT_V3_PARETO_50EP \
  > light_v3_pareto_50ep.log 2>&1 &

tail -f light_v3_pareto_50ep.log
```

该队列包含：Light03（激进轻量化）、Light04（Light03 + mask 质量排序）、Light07（G04 证据支持的 RepMixer + 官方 PAN 保守方案）。

## 5. PR/召回专项（仅在 Light03 精度有竞争力时）

```bash
nohup python -u 20260830_citrus_light_batch.py \
  --data /data/sxq/datasets/orange_yolo/data.yaml \
  --suite pr \
  --epochs 50 \
  --device 0 \
  --project /data/sxq/results/Light/CITRUS_LIGHT_V3_PR_50EP \
  > light_v3_pr_50ep.log 2>&1 &
```

LightP00--P03 使用完全相同的 Light03 结构，仅比较 BCE、VFL、NWD、VFL+NWD；LightP04 单独检验 mask-IoU 质量分支。标准 PR 图在最大可达召回之后补零，真正需要比较的是 recall ceiling、有效召回范围内 precision、原始 TP/FP/FN 和 Mask AP。

## 6. 最终复验

只有同时满足“精度不明显下降 + 实测速度/参数有优势”的候选才能进入 300 epoch。用 `--only` 明确模型，不要盲跑整个队列：

```bash
nohup python -u 20260830_citrus_light_batch.py \
  --data /data/sxq/datasets/orange_yolo/data.yaml \
  --suite final \
  --only Light03_deploy_lite \
  --epochs 300 \
  --seeds 42,43,44 \
  --device 0 \
  --project /data/sxq/results/Light/CITRUS_LIGHT_V3_FINAL_3SEED_300EP \
  > light_v3_final_3seed_300ep.log 2>&1 &
```

固定协议来自 `protocols/citrus_paper1_formal_v1.yaml`：AdamW、lr0=0.001、batch=16、imgsz=640、dropout=0、AMP=false、seed=42（最终 42/43/44）。若显式做 AMP 审计，必须单列 `AMP_AUDIT`，不能和正式表混算。

## 7. 单模型官方 YAML 入口

```python
from ultralytics import YOLO

model = YOLO("0_orange_yaml/Light_series/Light03_deploy_lite.yaml", task="segment")
model.load("yolo11n-seg.pt")
model.train(data="/data/sxq/datasets/orange_yolo/data.yaml", epochs=50, imgsz=640)
```

论文复现实验仍优先使用批量脚本，因为它会锁定超参数、记录源码/数据 YAML 快照并拒绝覆盖完成目录。

## 8. PR 和速度诊断

```bash
python analyze_citrus_pr.py \
  --weights /data/sxq/results/Light/CITRUS_LIGHT_V3_PARETO_50EP/Light03_deploy_lite/weights/best.pt \
  --data /data/sxq/datasets/orange_yolo/data.yaml \
  --device 0 --plots \
  --output-dir /data/sxq/results/Light/PR_DIAGNOSTICS

python profile_citrus_light.py \
  --device 0 --imgsz 640 --warmup 50 --repeats 200 --threads 1 \
  --output figures/citrus_light_profile_gpu.csv
```

延迟必须在同一张空闲 GPU、相同 batch=1、相同输入尺寸下依次测量；训练过程中测出的延迟不能写进论文。
