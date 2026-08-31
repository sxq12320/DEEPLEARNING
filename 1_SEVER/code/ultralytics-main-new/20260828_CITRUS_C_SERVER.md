# C 系列服务器训练说明

假定整个 `1_SEVER/code` 已上传到 `/data/sxq/code`，清洗后的数据配置位于：

```text
/data/sxq/datasets/orange_yolo_2/data.yaml
```

进入代码并安装当前修改版：

```bash
cd /data/sxq/code/ultralytics-main-new
pip install -e .
```

先做 3 epoch smoke：

```bash
python -u 20260828_citrus_c_batch.py \
  --data /data/sxq/datasets/orange_yolo_2/data.yaml \
  --suite smoke --epochs 3 --batch 16 --workers 4 --device 0 \
  --project 1_results/C_series/CITRUS_C_SMOKE_3EP
```

推荐先跑三个高优先级新结构：

```bash
nohup python -u 20260828_citrus_c_batch.py \
  --data /data/sxq/datasets/orange_yolo_2/data.yaml \
  --suite core --only C03_dualproto_core,C04_dualproto_hwd,C07_dualproto_context \
  --epochs 50 --batch 16 --workers 4 --device 0 \
  --project 1_results/C_series/CITRUS_C_PRIORITY_50EP \
  > c_priority_50ep.log 2>&1 &

tail -f c_priority_50ep.log
```

若要把所有 C 改进和历史控制一次跑 300 epoch（脚本中没有原版 YOLO11）：

```bash
nohup python -u 20260828_citrus_c_batch.py \
  --data /data/sxq/datasets/orange_yolo_2/data.yaml \
  --suite architectures --epochs 300 --batch 16 --workers 4 --device 0 \
  --project 1_results/C_series/CITRUS_C_ALL_300EP \
  > c_all_300ep.log 2>&1 &

tail -f c_all_300ep.log
```

结构胜出后再跑 loss 消融，避免结构和损失同时变化：

```bash
nohup python -u 20260828_citrus_c_batch.py \
  --data /data/sxq/datasets/orange_yolo_2/data.yaml \
  --suite losses --epochs 300 --batch 16 --workers 4 --device 0 \
  --project 1_results/C_series/CITRUS_C_LOSS_300EP \
  > c_loss_300ep.log 2>&1 &
```

不要让 C 与 L 系列同时占用同一张 GPU。若有两张独立 GPU，可把另一条命令设为 `--device 1`。

## L 系列（排除原版 A00）

L 系列是早于 B/C 的 CitrusTopo 探索，优先级低于 C03/C04/C07。若仍需完整补实验，可运行：

算力紧张时，建议只补四个具有解释价值的单因素/双因素模型：

```bash
nohup python -u 20260824_citrus_topo_batch.py \
  --data /data/sxq/datasets/orange_yolo_2/data.yaml \
  --suite architectures --only A01_lska,A03_p2cfs,A04_topology,A06_lska_topology \
  --epochs 300 --batch 16 --workers 4 --device 0 \
  --project 1_results/L_series/CITRUS_L_PRIORITY_300EP \
  > l_priority_300ep.log 2>&1 &

tail -f l_priority_300ep.log
```

若确实要补齐全部 9 个改进模型，再运行：

```bash
nohup python -u 20260824_citrus_topo_batch.py \
  --data /data/sxq/datasets/orange_yolo_2/data.yaml \
  --suite architectures \
  --only A01_lska,A02_scale,A03_p2cfs,A04_topology,A05_lska_scale,A06_lska_topology,A07_full_core,A08_scale_topology,A09_full_p2cfs \
  --epochs 300 --batch 16 --workers 4 --device 0 \
  --project 1_results/L_series/CITRUS_L_ARCH_300EP \
  > l_series_300ep.log 2>&1 &

tail -f l_series_300ep.log
```

这条命令会按顺序训练 9 个改进模型，不训练 `A00_reference`。L 系列 loss 权重较旧且偏大，不建议在架构筛选前把 `--suite losses` 一起跑完。
