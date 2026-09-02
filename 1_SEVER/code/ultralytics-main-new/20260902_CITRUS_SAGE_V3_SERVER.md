# SAGE-v3 服务器运行指南

_最后验证：2026-09-02 · Python 3.9 兼容_

---

## 先做什么

上传整个 `1_SEVER/code` 后，只进入当前活动代码库：

```bash
cd /data/sxq/code/ultralytics-main-new
pip install -e .
```

以下命令假设数据 YAML 是：

```bash
DATA=/data/sxq/datasets/orange_yolo/data.yaml
```

脚本不会替你寻找或改写数据路径，只读取你传入的 `--data`。

## 1. 无训练构建检查

```bash
python -u 20260902_citrus_sage_v3_batch.py \
  --data "$DATA" \
  --suite all \
  --epochs 50 \
  --dry-run
```

应看到 8 行 `BUILD OK`。这一步不占用 300 epoch 训练时间。

## 2. 目标 GPU 速度门

确认 GPU 空闲后依次执行：

```bash
python benchmark_citrus_sage.py \
  --model SAGE20_shape_context_backbone.yaml \
  --device 0 --imgsz 640 --batch 2 --warmup 10 --iterations 30 --max-ratio 1.20

python benchmark_citrus_sage.py \
  --model SAGE21_innovation_pyramid.yaml \
  --device 0 --imgsz 640 --batch 2 --warmup 10 --iterations 30 --max-ratio 1.20

python benchmark_citrus_sage.py \
  --model SAGE23_joint_core_v3.yaml \
  --device 0 --imgsz 640 --batch 2 --warmup 10 --iterations 30 --max-ratio 1.20
```

任何候选显示 `SPEED GATE FAILED`，都不要直接长训。先保存输出并反馈结果。

## 3. 1–3 epoch 冒烟训练

```bash
nohup python -u 20260902_citrus_sage_v3_batch.py \
  --data "$DATA" \
  --suite smoke \
  --epochs 3 \
  --device 0 \
  --project /data/sxq/results/SAGE/CITRUS_SAGE_V3_SMOKE_3EP \
  > sage_v3_smoke_3ep.log 2>&1 &
```

查看日志：

```bash
tail -n 80 sage_v3_smoke_3ep.log
tail -f sage_v3_smoke_3ep.log
```

默认 smoke 只跑 SAGE20、SAGE21、SAGE23。确认无 NaN、CUDA error、显存持续上涨或离谱吞吐后再继续。

## 4. 默认 50 epoch 结构筛选

```bash
nohup python -u 20260902_citrus_sage_v3_batch.py \
  --data "$DATA" \
  --suite screen \
  --epochs 50 \
  --device 0 \
  --project /data/sxq/results/SAGE/CITRUS_SAGE_V3_SCREEN_50EP \
  > sage_v3_screen_50ep.log 2>&1 &
```

默认依次运行：

1. SAGE20：主干单变量；
2. SAGE21：融合单变量；
3. SAGE22：显式拓扑监督；
4. SAGE23：联合核心。

SAGE24–27 不在默认队列中。只有 SAGE23 通过 50 epoch 门槛后，才需要进一步测试颜色统计、PR 排序和
遮挡损失。

## 5. 运行后续三个机制消融

```bash
nohup python -u 20260902_citrus_sage_v3_batch.py \
  --data "$DATA" \
  --suite all \
  --only SAGE24_style_robust,SAGE25_quality_aligned,SAGE26_occlusion_topology \
  --epochs 50 \
  --device 0 \
  --project /data/sxq/results/SAGE/CITRUS_SAGE_V3_MECHANISM_50EP \
  > sage_v3_mechanism_50ep.log 2>&1 &
```

## 6. 最终 300 epoch 三种子

把 `--only` 改为 50 epoch 的真实优胜者。下面仅示范 SAGE23，并不预先宣布它最好：

```bash
nohup python -u 20260902_citrus_sage_v3_batch.py \
  --data "$DATA" \
  --suite final \
  --only SAGE23_joint_core_v3 \
  --epochs 300 \
  --seeds 42,43,44 \
  --device 0 \
  --project /data/sxq/results/SAGE/CITRUS_SAGE_V3_FINAL_300EP \
  > sage_v3_final_300ep.log 2>&1 &
```

## 固定超参数说明

批量脚本读取 `protocols/citrus_paper1_formal_v1.yaml`，固定 batch、imgsz、optimizer、学习率、增强、
dropout、mask ratio、seed 和 AMP。当前正式协议锁定 `amp=false`。如果原基线使用 `amp=true`，它与本批
结果不是严格的结构对照；应使用相同 AMP 重新做一个配对基线，或显式运行 `--amp` 并把结果标记为 AMP
审计，不能混入正式架构表。

## 如何停止

先找进程：

```bash
pgrep -af 20260902_citrus_sage_v3_batch.py
```

再对准确 PID 发送正常终止：

```bash
kill PID
```

等待数秒后仍未退出，再检查：

```bash
ps -p PID -o pid,stat,etime,cmd
```

不要使用模糊的全局 `pkill python`，否则可能停止其他人的训练。

## 结果判定

至少同时比较：mask mAP50-95、mask mAP50、precision、recall、AP-tiny、Params、GFLOPs、GPU
step time、伪装子集 AP、低 solidity 子集 AP、相邻实例 split/merge error。PR 曲线高召回端下降本身不是
程序错误；只有 SAGE25 在同协议下提升固定高召回点的 precision，才支持“排序校准有效”的结论。
