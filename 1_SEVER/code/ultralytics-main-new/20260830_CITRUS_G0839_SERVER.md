# G_0839 服务器运行说明

正式固定超参数见 `CITRUS_FORMAL_PROTOCOL.md`，机器可读版本为 `protocols/citrus_paper1_formal_v1.yaml`。训练入口会拒绝不同的 batch、imgsz、workers、cache 和未声明 AMP 变化。

## 1. 安装当前代码

```bash
cd /data/sxq/code/ultralytics-main-new
pip install -e .
```

不要从另一个 `ultralytics` 目录启动。可先确认导入位置：

```bash
python -c "import ultralytics; print(ultralytics.__file__)"
```

输出路径必须指向 `/data/sxq/code/ultralytics-main-new/ultralytics/`。

## 2. 只检查六个模型能否构建

数据路径完全由你传入，脚本不会再做指纹确认或替你选择数据集：

```bash
python 20260830_citrus_g0839_batch.py \
  --data /你的实际数据集/data.yaml \
  --suite all \
  --dry-run
```

## 3. 先跑 3 epoch smoke

```bash
nohup python -u 20260830_citrus_g0839_batch.py \
  --data /你的实际数据集/data.yaml \
  --suite smoke \
  --epochs 3 \
  --batch 16 \
  --workers 4 \
  --device 0 \
  --project /data/sxq/results/G_0839/CITRUS_G0839_SMOKE_3EP \
  > g0839_smoke.log 2>&1 &
```

查看日志：

```bash
tail -f g0839_smoke.log
```

## 4. 再做 50 epoch 结构筛选

```bash
nohup python -u 20260830_citrus_g0839_batch.py \
  --data /你的实际数据集/data.yaml \
  --suite screen \
  --epochs 50 \
  --batch 16 \
  --workers 4 \
  --device 0 \
  --project /data/sxq/results/G_0839/CITRUS_G0839_SCREEN_50EP \
  > g0839_screen.log 2>&1 &
```

## 5. 只把筛选胜出的模型跑 300 epoch

下面只是命令模板，不能在 50 epoch 结果出来前预设 G04/G05 一定胜出：

```bash
nohup python -u 20260830_citrus_g0839_batch.py \
  --data /你的实际数据集/data.yaml \
  --only G04_boundary_refine,G05_full_sdr \
  --epochs 300 \
  --batch 16 \
  --workers 4 \
  --device 0 \
  --project /data/sxq/results/G_0839/CITRUS_G0839_FINAL_300EP \
  > g0839_final.log 2>&1 &
```

最终三个种子使用 `--seeds 42,43,44`。脚本逐个模型、逐个种子顺序训练，不会在同一张卡上并发。

## 6. 单独训练任意 YAML

```bash
python train_citrus_yaml.py \
  --model 0_orange_yaml/G_0839_series/05_g05_full_sdr.yaml \
  --name G05_full_sdr_seed42 \
  --data /你的实际数据集/data.yaml \
  --project /data/sxq/results/G_0839/CITRUS_G0839_SINGLE_300EP \
  --epochs 300 \
  --batch 16 \
  --workers 4 \
  --device 0
```

单模型入口会按照 YAML 中的 `sdr_stage` 自动启用与批量脚本相同的 loss；显式命令行参数仍可覆盖默认值。

## 7. 停止实验

```bash
pgrep -af 20260830_citrus_g0839_batch.py
kill <上一步显示的PID>
```

先用 `kill` 正常终止并等待保存；只有进程长期不退出时才使用 `kill -9 <PID>`。

## 协议提醒

- G00 是轻量头控制组，不是官方 YOLO11n-seg 基线；正式论文仍须引用相同划分和协议下的官方基线结果。
- 默认关闭 AMP，与现有 S/B grouped-clean 结果保持一致。`--amp` 只能用于同模型、同数据、同 seed 的成对 AMP 对照，不能与 AMP-off 结果混合归因。
- 50 epoch 不能只看 mAP50；至少同时看 mask mAP50-95、recall、AP_small、低 solidity、near-gap 和 split/merge。

## AMP 成对审计

如果需要验证 AMP 是否改变精度，只跑 G00/G01 两个控制模型即可，并且除 AMP 外不允许改变任何设置：

```bash
# AMP off（正式协议）
python 20260830_citrus_g0839_batch.py \
  --data /你的实际数据集/data.yaml \
  --only G00_lite_control,G01_preserve --epochs 50 --seeds 42 \
  --project /data/sxq/results/G_0839/AMP_AUDIT_OFF

# AMP on（唯一变化）
python 20260830_citrus_g0839_batch.py \
  --data /你的实际数据集/data.yaml \
  --only G00_lite_control,G01_preserve --epochs 50 --seeds 42 --amp \
  --project /data/sxq/results/G_0839/AMP_AUDIT_ON
```
