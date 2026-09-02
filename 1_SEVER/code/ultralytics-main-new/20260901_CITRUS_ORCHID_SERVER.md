# ORCHID 服务器运行说明

## 1. 进入唯一代码目录并安装当前 fork

```bash
cd /data/sxq/code/ultralytics-main-new
pip install -e .
```

不要从同级另一个 `ultralytics` 目录运行。确认脚本与 YAML 都存在：

```bash
ls -l 20260901_citrus_orchid_batch.py
ls -l 0_orange_yaml/ORCHID_series/*.yaml
```

## 2. 只构建，不训练

数据路径由你明确传入，脚本不会猜路径，也没有指纹确认门槛：

```bash
python -u 20260901_citrus_orchid_batch.py \
  --data /data/sxq/datasets/orange_yolo/data.yaml \
  --suite all \
  --epochs 300 \
  --dry-run
```

应该看到 ORCHID01--06 全部输出 `BUILD OK`。

## 3. 必须先做 3 epoch 烟雾测试

```bash
python -u 20260901_citrus_orchid_batch.py \
  --data /data/sxq/datasets/orange_yolo/data.yaml \
  --suite smoke \
  --epochs 3 \
  --device 0 \
  --project /data/sxq/results/ORCHID/CITRUS_ORCHID_SMOKE_3EP
```

Smoke 只跑 ORCHID03 和 ORCHID04，用于排除数据、损失、CUDA 和反传问题。

## 4. 推荐的 50 epoch 结构筛选

```bash
nohup python -u 20260901_citrus_orchid_batch.py \
  --data /data/sxq/datasets/orange_yolo/data.yaml \
  --suite screen \
  --epochs 50 \
  --device 0 \
  --project /data/sxq/results/ORCHID/CITRUS_ORCHID_SCREEN_50EP \
  > orchid_screen_50ep.log 2>&1 &
```

```bash
tail -f orchid_screen_50ep.log
```

## 5. 如果你坚持将全部新模型直接跑 300 epoch

`--suite all` 只包含 ORCHID01--06，不会重复旧基线：

```bash
nohup python -u 20260901_citrus_orchid_batch.py \
  --data /data/sxq/datasets/orange_yolo/data.yaml \
  --suite all \
  --epochs 300 \
  --device 0 \
  --project /data/sxq/results/ORCHID/CITRUS_ORCHID_ALL_300EP \
  > orchid_all_300ep.log 2>&1 &
```

```bash
tail -f orchid_all_300ep.log
```

脚本会在同一张 GPU 上逐个训练，不能并发占满同一张卡。正式协议固定 `batch=16`、`workers=4`、`imgsz=640`、`AdamW`、`lr0=0.001`、`dropout=0`、`amp=false`。不要为了快而悄悄修改这些值，否则与现有结果不可比。

## 6. 单独训练某一个模型

```bash
nohup python -u 20260901_citrus_orchid_batch.py \
  --data /data/sxq/datasets/orange_yolo/data.yaml \
  --suite final \
  --only ORCHID03_supervised_query_router \
  --epochs 300 \
  --device 0 \
  --project /data/sxq/results/ORCHID/CITRUS_ORCHID_FINAL_300EP \
  > orchid03_300ep.log 2>&1 &
```

最终论文三种子：

```bash
python -u 20260901_citrus_orchid_batch.py \
  --data /data/sxq/datasets/orange_yolo/data.yaml \
  --suite final \
  --only ORCHID03_supervised_query_router \
  --epochs 300 \
  --seeds 42,43,44 \
  --device 0 \
  --project /data/sxq/results/ORCHID/CITRUS_ORCHID03_3SEED_300EP
```

## 7. 停止与检查

```bash
jobs -l
ps -ef | grep 20260901_citrus_orchid_batch.py | grep -v grep
kill <PID>
```

先使用普通 `kill`；只有进程无法退出时才使用 `kill -9 <PID>`。
