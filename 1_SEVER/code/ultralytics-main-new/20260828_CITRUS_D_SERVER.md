# CitrusD 服务器训练说明

假设上传后的工程目录是：

```text
/data/sxq/1_SEVER/code/ultralytics-main-new
```

数据集路径不写死在模型 YAML 中。每次通过 `--data` 指向服务器上的清洗数据 `data.yaml`；脚本会在结果目录生成运行期 YAML，只修正根路径，不改变 train/val/test 成员。

## 1. 安装当前定制版本

```bash
cd /data/sxq/1_SEVER/code/ultralytics-main-new
pip install -e .
```

D 系列不需要 Mamba、MMCV 或 `pytorch_wavelets`。确认导入的是当前目录：

```bash
python -c "import ultralytics; print(ultralytics.__file__)"
python profile_citrus_d.py
python -m pytest tests/test_citrus_d.py -q
```

## 2. 先做 3 epoch smoke

把下面的 `/data/sxq/datasets/orange_clean/data.yaml` 换成服务器真实路径：

```bash
python -u 20260828_citrus_d_batch.py \
  --data /data/sxq/datasets/orange_clean/data.yaml \
  --suite smoke --epochs 3 --batch 16 --workers 4 --device 0 \
  --project 1_results/D_series/CITRUS_D_SMOKE_3EP
```

smoke 必须九个模型都能生成 `weights/last.pt`、`results.csv`，再开始正式筛选。

## 3. 推荐的 50 epoch 因果筛选

先验证主干设计中的三个因果问题：PDC、深层语义门控、无色结构 stem。

```bash
nohup python -u 20260828_citrus_d_batch.py \
  --data /data/sxq/datasets/orange_clean/data.yaml \
  --suite controls --epochs 50 --batch 16 --workers 4 --device 0 \
  --project 1_results/D_series/CITRUS_D_CONTROLS_50EP \
  > d_controls_50ep.log 2>&1 &

tail -f d_controls_50ep.log
```

然后在同一协议下跑完整候选：

```bash
nohup python -u 20260828_citrus_d_batch.py \
  --data /data/sxq/datasets/orange_clean/data.yaml \
  --suite core --epochs 50 --batch 16 --workers 4 --device 0 \
  --project 1_results/D_series/CITRUS_D_CORE_50EP \
  > d_core_50ep.log 2>&1 &

tail -f d_core_50ep.log
```

脚本在单张卡上**严格串行**运行，完成一个模型、清理显存后才开始下一个。不要再在同一 GPU 上启动 C/L/D 的第二个训练进程。

## 4. 300 epoch

推荐只把 50 epoch 的前两名跑到 300。例：

```bash
nohup python -u 20260828_citrus_d_batch.py \
  --data /data/sxq/datasets/orange_clean/data.yaml \
  --suite core --only D06_shape_semantic_full,D07_deploy_lite \
  --epochs 300 --batch 16 --workers 4 --device 0 \
  --project 1_results/D_series/CITRUS_D_FINALISTS_300EP \
  > d_finalists_300ep.log 2>&1 &

tail -f d_finalists_300ep.log
```

如果确实要一次跑完九个改进模型（不含原版 YOLO11）：

```bash
nohup python -u 20260828_citrus_d_batch.py \
  --data /data/sxq/datasets/orange_clean/data.yaml \
  --suite architectures --epochs 300 --batch 16 --workers 4 --device 0 \
  --project 1_results/D_series/CITRUS_D_ALL_300EP \
  > d_all_300ep.log 2>&1 &

tail -f d_all_300ep.log
```

## 5. 中断、恢复与检查

查进程：

```bash
ps -ef | grep 20260828_citrus_d_batch.py | grep -v grep
nvidia-smi
```

正常停止当前批处理：

```bash
kill <PID>
```

用完全相同的命令重启时，脚本会：

- 跳过有完成记录的模型；
- 对存在 `weights/last.pt` 的当前模型执行 Ultralytics `resume=True`；
- 不覆盖已完成实验。

快速看最后日志：

```bash
tail -n 100 d_core_50ep.log
tail -f d_core_50ep.log
```

## 6. 协议锁定值

正式对照固定：`imgsz=640`、AdamW、`lr0=0.001`、`weight_decay=0.0005`、`dropout=0`、`amp=False`、seed 42、batch 16。脚本已写入这些值。若显存不足只降低 `--batch`；不要在一组比较中途改变图像尺寸、优化器、AMP、数据划分或增强参数。

