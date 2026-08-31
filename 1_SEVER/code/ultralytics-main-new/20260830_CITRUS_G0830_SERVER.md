# G_0830 服务器运行说明

进入你上传后的活动 fork：

```bash
cd /data/sxq/code/ultralytics-main-new
pip install -e .
```

先确认脚本与数据路径：

```bash
python 20260830_citrus_g0830_batch.py \
  --data /data/sxq/datasets/orange_yolo/data.yaml \
  --suite structure --epochs 50 --device 0 --dry-run
```

先做 3 epoch smoke：

```bash
python 20260830_citrus_g0830_batch.py \
  --data /data/sxq/datasets/orange_yolo/data.yaml \
  --suite smoke --epochs 3 --batch 16 --workers 4 --device 0 \
  --project /data/sxq/results/G_0830/CITRUS_G0830_SMOKE_3EP
```

随后做结构筛选。推荐先 50 epoch；若你坚持直接 300 epoch，第三条命令已给出。批量脚本严格串行，一张卡不会同时启动两个训练。

```bash
nohup python -u 20260830_citrus_g0830_batch.py \
  --data /data/sxq/datasets/orange_yolo/data.yaml \
  --suite structure --epochs 50 --batch 16 --workers 4 --device 0 \
  --project /data/sxq/results/G_0830/CITRUS_G0830_STRUCTURE_50EP \
  > g0830_structure_50ep.log 2>&1 &

nohup python -u 20260830_citrus_g0830_batch.py \
  --data /data/sxq/datasets/orange_yolo/data.yaml \
  --suite all --epochs 300 --batch 16 --workers 4 --device 0 \
  --project /data/sxq/results/G_0830/CITRUS_G0830_ALL_300EP \
  > g0830_all_300ep.log 2>&1 &
```

查看进度：

```bash
tail -f g0830_structure_50ep.log
# 或
tail -f g0830_all_300ep.log
```

停止时先找 PID，再只终止这个批量脚本及其子训练进程：

```bash
pgrep -af 20260830_citrus_g0830_batch.py
kill -TERM <PID>
```

继续同一批次时可使用 `--skip-completed` 并保持相同 project；脚本会跳过同时存在 `best.pt` 和 `results.csv` 的已完成目录。若某个目录是失败后留下的不完整 run，脚本会拒绝覆盖；请保留它作审计并换一个新的 `--project`，或确认不再需要后由你手动归档该失败目录。

固定协议：AdamW、lr0=0.001、lrf=0.01、imgsz=640、batch=16、workers=4、dropout=0、AMP=False、seed=42、patience=300。数据路径由 `--data` 直接指定，没有指纹确认门。
