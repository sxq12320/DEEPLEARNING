# SAGE V6 今晚怎么跑

## 上传与前台入口

上传更新后的整个 `1_SEVER/code/ultralytics-main-new` 代码目录。不能只上传RUN或YAML：本次也修改了模块导出、tasks解析/权重映射和前台注册。保留服务器已有数据，不覆盖结果。不要用pip升级替换本地定制ultralytics。

在服务器VS Code选择训练环境，打开 `RUN_SAGE_V6.py`，修改顶部 `DATA` 和 `DEVICE`。数据填写你自己的新清洗数据yaml路径；不会重分割、删除或清洗数据。默认设置：

```python
DATA = "/data/sxq/datasets/orange_yolo/data.yaml"
DEVICE = "1"
SUITE = "structure"
EPOCHS = 300
DRY_RUN = False
```

点击右上角运行三角形，或终端前台运行：

```bash
cd /data/sxq/code/ultralytics-main-new
/home/amax/sxq/bin/python RUN_SAGE_V6.py
```

默认逐个跑60–64五个模型，固定seed42，结果在 `/data/sxq/results/SAGE/CITRUS_SAGE_V6_STRUCTURE_300EP`。没有nohup、不同时启动多个模型。队列顺序会固定随机化，非编号顺序。Ctrl+C停止整个队列，不启动下一模型。已完成实验不覆盖；半成品需另起项目目录，不会擅自接续。

`SUITE="all"` 加上65，共六个；`"priority"` 只跑60/62/64。65增加几何损失，本地训练步开销约多20%，建议结构结果出来后再选。前台启动并不天然消除资源竞争；本程序额外使用单卡锁及忙卡检查。

## 先做服务器兼容与短训练检查

本地验证环境为Python3.9/Torch2.8 CPU，并未验证你服务器的GPU/Torch1.13环境。先运行：

```bash
python -c "import sys,ultralytics,torch; print(sys.executable); print(ultralytics.__file__); print(torch.__version__)"
python 20260905_citrus_sage_v6_batch.py --data /data/sxq/datasets/orange_yolo/data.yaml --suite all --dry-run
python 20260905_citrus_sage_v6_batch.py --data /data/sxq/datasets/orange_yolo/data.yaml --suite all --epochs 1 --device 1 --project /data/sxq/results/SAGE/CITRUS_SAGE_V6_SERVER_SMOKE_1EP --fail-fast
```

确认import来自当前代码目录；`--dry-run`只构建，不训练，不要把BUILD OK误认为已经开跑。短训练和正式训练必须不同项目目录。使用同一个解释器执行上述命令。如果重复测试，请换新输出目录。

可在GPU空闲、正式训练前测同卡算子吞吐：

```bash
python scripts/benchmark_sage_v6.py --device 1 --batch 16 --imgsz 640 --steps 20 --output reports/sage_v6_gpu_server.json
```

该合成微测不含数据加载/优化器/NMS，不等同于真实一轮时间。若显存不足，停止并记录，不要静默只改某个模型的正式batch。

## 固定超参数

完整真源：`protocols/citrus_paper1_formal_v1.yaml`。沿用V5，不随模型调参：AMP=false、batch16、imgsz640、workers4、AdamW、lr0=.001、lrf=.01、weight_decay=.0005、dropout=0、mask_ratio4、overlap_mask=true、patience300、close_mosaic10、seed42；增强和验证设置也统一。65只有显式方法损失差异。训练保存协议、源码hash、数据加载名单和初始化记录。

## 单模型官方API

六个YAML均在 `0_orange_yaml/SAGE_V6_series`，已直接支持YOLO构建，不依赖先运行批量入口来注册模块。若只是单独跑64，也可以设置入口 `ONLY="SAGE64_selective_exchange"`，保留完整审计与best_mask保存。

标准API示例（另存Python文件，用三角形运行）：

```python
import torch
from ultralytics import YOLO
from citrus_protocol import fixed_train_args, load_protocol

if __name__ == "__main__":
    torch.manual_seed(42)
    model = YOLO("0_orange_yaml/SAGE_V6_series/SAGE64_selective_exchange.yaml", task="segment")
    model.load("yolo11n-seg.pt")
    args = fixed_train_args()
    args.update(load_protocol()["fixed_validation"])
    model.train(**args, data="/data/sxq/datasets/orange_yolo/data.yaml",
                epochs=300, device=1, seed=42,
                project="/data/sxq/results/SAGE/V6_SINGLE_NEW", name="SAGE64", exist_ok=False)
```

请在代码根目录运行。标准API示例不包含批量入口的初始化审计/专用best_mask回调，论文比较更推荐ONLY方式。不要用默认model.train参数与固定协议批量结果直接比较。
