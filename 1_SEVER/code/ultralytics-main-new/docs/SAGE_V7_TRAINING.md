# SAGE V7 运行说明

## 上传和点击运行

上传这次更新后的整个 `1_SEVER/code/ultralytics-main-new` 代码目录；不能只上传RUN或YAML，因为包含新头部、模块导出和tasks注册。保留原有结果和服务器数据，不使用pip升级覆盖定制库。无需安装Mamba、mmcv或spconv。

在服务器VS Code选择训练环境，打开 `RUN_SAGE_V7.py`，修改：

```python
DATA = "/data/sxq/datasets/orange_yolo/data.yaml"  # 你的清洗数据yaml
DEVICE = "1"                                # 你要用的空闲GPU
SUITE = "structure"                         # 70--74共五个
EPOCHS = 300
DRY_RUN = False
```

右上角三角形直接运行，或者：

```bash
cd /data/sxq/code/ultralytics-main-new
/home/amax/sxq/bin/python RUN_SAGE_V7.py
```

当前进程前台、单GPU锁、逐模型串行，Ctrl+C停止整个队列，不使用nohup。模型顺序固定随机化；已完成不重跑、不覆盖半成品。默认结果 `/data/sxq/results/SAGE/CITRUS_SAGE_V7_STRUCTURE_300EP`。70是60的复现对照，不是又加一个不同官方基线。

省预算可以 `SUITE="priority"` 跑70/71/72；`ONLY="SAGE74_p2_local_context"` 可单独跑模块版本（建议换独立PROJECT）。所有设置应在开始前改好，正在跑时不要修改源码。

## 先验收服务器兼容和速度

本地验证为Python3.9/Torch2.8 CPU，不等于你的服务器环境已经验证。用同一解释器检查路径、构建，随后独立1轮短训练：

```bash
python -c "import sys,torch,ultralytics; print(sys.executable); print(torch.__version__); print(ultralytics.__file__)"
python 20260906_citrus_sage_v7_batch.py --data /data/sxq/datasets/orange_yolo/data.yaml --suite all --dry-run
python 20260906_citrus_sage_v7_batch.py --data /data/sxq/datasets/orange_yolo/data.yaml --suite all --epochs 1 --device 1 --project /data/sxq/results/SAGE/CITRUS_SAGE_V7_SMOKE_1EP --fail-fast
```

确认import来自当前代码目录。`--dry-run`只构建退出，没有训练。短训练输出必须与正式300轮不同；重复短训练换新目录。确认GPU没有旧任务后启动，不同时开两个队列。

同卡空闲时可先测算子成本：

```bash
python scripts/benchmark_sage_v7.py --device 1 --batch 16 --imgsz 640 --steps 20 --output reports/sage_v7_gpu_server.json
```

这是合成微基准，不含DataLoader/优化器/NMS。真正的一轮耗时、显存和验证耗时仍要看短训练。若74等版本慢很多，先暂停该版本，保留70/71/72对照，不为跑完队列硬耗算力。前台本身不会消除数据读取或其他用户竞争。

## 固定超参数与日志

继续沿用 `protocols/citrus_paper1_formal_v1.yaml`：AMP=false、batch16、imgsz640、workers4、AdamW lr0=.001/lrf=.01/weight_decay=.0005、dropout0、mask_ratio4、overlap_mask=true、seed42、300轮。增强、损失权重和评估设置不随模型改动。三种子仅最终确认阶段选择42/43/44。

实时输出在VS Code终端；每模型目录有results.csv、args.yaml、weights/best.pt、weights/best_mask.pt、初始化审计和加载样本记录。best_mask按Mask AP50–95选择，官方best.pt保持原规则。不要把fixture短训练AP当结果。

如果要用tail查看持续写入的数值：

```bash
tail -n 5 /data/sxq/results/SAGE/CITRUS_SAGE_V7_STRUCTURE_300EP/SAGE72_p2_candidates_seed42/results.csv
```

## 单模型官方入口

五个YAML均在 `0_orange_yaml/SAGE_V7_series`，不依赖批量脚本临时注册。可以 `YOLO(yaml).load("yolo11n-seg.pt").train(...)`，但须传入固定协议，不能用默认参数与正式实验混比。建议在RUN里设置ONLY，保留审计、锁和专用best_mask回调。新头本轮没有增加额外损失函数。

旧V6的62仅上传了118轮；不要把它当300轮完成，也不要让新代码覆盖它的原输出。
