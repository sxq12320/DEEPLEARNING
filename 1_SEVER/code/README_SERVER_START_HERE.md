# 服务器开始入口（CitrusB v2，2026-08-27）

上传整个 `1_SEVER/code` 后进入真正的项目根目录：

```bash
cd /你的服务器路径/code/ultralytics-main-new
```

外层 `ultralytics-main-new/` 是项目，内层 `ultralytics/` 是 Python 包，并不是两套重复代码。B v2 不
依赖 Mamba、MMCV 或自定义 CUDA 算子。

## 1. 环境自检

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
pip install -e .
python -m pytest -q tests/test_citrus_b.py tests/test_citrus_b_report.py
```

`torch.cuda.is_available()` 必须为 `True`。若需要新建环境：

```bash
conda env create -f environment_citrus_b.yml
conda activate citrus-b
pip install -e .
```

## 2. 数据路径

训练命令的 `--data` 必须指向服务器上的清洗数据，例如：

```text
/data/sxq/datasets/orange_yolo_grouped_dedup_20260820/data.yaml
```

批处理脚本会生成服务器安全的运行时 YAML，不改变已有 train/val/test 成员。先确认路径：

```bash
python -c "from pathlib import Path; p=Path('/data/sxq/datasets/orange_yolo_grouped_dedup_20260820/data.yaml'); print(p.resolve(), p.is_file())"
```

## 3. 现在先做：B02-B09 三 epoch 冒烟

S00 和 S04 已在完全相同协议下分别覆盖 B00、B01。先只运行八个新结构，检查数据加载、前向、反向、
验证和权重保存：

```bash
python 20260826_citrus_b_batch.py \
  --data /data/sxq/datasets/orange_yolo_grouped_dedup_20260820/data.yaml \
  --suite smoke --epochs 3 --batch 4 --workers 4 --device 0 \
  --project 1_results/CITRUS_B_V2_SMOKE
```

先预览而不训练可加 `--dry-run`。冒烟指标不用于比较精度。

## 4. 冒烟正常后：含同周期基线的 50 epoch 结构筛选

```bash
nohup python 20260826_citrus_b_batch.py \
  --data /data/sxq/datasets/orange_yolo_grouped_dedup_20260820/data.yaml \
  --suite screening --epochs 50 --batch 16 --workers 4 --device 0 \
  --project 1_results/CITRUS_B_V2_SCREEN_50EP \
 > citrus_b_v2_screen_50ep.log 2>&1 &
```

这里会运行 B00-B09。不能拿 50-epoch 的新模型直接对比已经训练 300 epoch 的 S00/S04；因此 B00、
B01 必须在 50-epoch 筛选中同周期重跑，作为公平参考。

```bash
tail -f citrus_b_v2_screen_50ep.log
```

完成后汇总：

```bash
python report_citrus_b_results.py --project 1_results/CITRUS_B_V2_SCREEN_50EP
python benchmark_citrus_b_latency.py --device 0 --warmup 30 --iterations 200
```

筛选时同时看 Mask AP50-95、Mask AP50、recall 和同机 GPU latency。B08 偏严格 AP，B09 偏召回/
速度；名称不预设最终胜者。

## 5. 何时跑 300 epoch

只把 50-epoch 中满足以下条件的 1-2 个模型扩展到 300 epoch：

- 在 50-epoch 同周期比较中明显超过 B01，并在 300-epoch 后再与 S04 的 0.6150 比较；
- Mask AP50 和 recall 没有出现 S09 式明显回退；
- 服务器实测延迟和显存可接受。

示例（把名称替换成真实入选模型）：

```bash
nohup python 20260826_citrus_b_batch.py \
  --data /data/sxq/datasets/orange_yolo_grouped_dedup_20260820/data.yaml \
  --suite architectures --epochs 300 --batch 16 --workers 4 --device 0 \
  --only B02_context_lite,B09_recall_balanced \
  --project 1_results/CITRUS_B_V2_FINAL_300EP \
  > citrus_b_v2_final_300ep.log 2>&1 &
```

最终入选模型与 B00 基线再使用 `--seeds 42,43,44` 做三次重复。不要在结构尚未筛出之前运行
`--suite losses`；损失套件是第二阶段实验。

## 6. 单独训练 B09

标准 YAML 入口仍然可用，但必须显式传入已选定的 B/Q 损失：

```bash
python train_citrus_yaml.py \
  --model 0_orange_yaml/20260826_citrus_b/09_b09_recall_balanced_final.yaml \
  --data /data/sxq/datasets/orange_yolo_grouped_dedup_20260820/data.yaml \
  --name B09_single --epochs 300 --batch 16 --device 0 \
  --citrus-boundary 0.25 --citrus-query 0.05
```

完整 S 结果、B v2 取舍和资源表见
`ultralytics-main-new/20260826_CITRUS_B_EVIDENCE_AND_DESIGN.md`。
