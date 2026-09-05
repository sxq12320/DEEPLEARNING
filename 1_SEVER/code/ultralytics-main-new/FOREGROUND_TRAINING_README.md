# VS Code 前台串行训练入口

统一入口是 `RUN_CITRUS_FOREGROUND.py`。SAGE V4 重构版也提供更不易选错系列的专用入口 `RUN_SAGE_V4.py`。
它用于替代 `nohup ... &`，但不替代各系列原有的实验定义。
程序直接在当前 Python 进程中调用对应批量脚本，因此 VS Code 终端会实时显示日志。在运行它的终端按一次
`Ctrl+C` 会退出队列，不再启动下一模型；VS Code 强制停止可能跳过正常清理步骤。

## 使用方法

1. 在服务器 VS Code 中打开 `RUN_CITRUS_FOREGROUND.py`。
2. 只修改文件顶部 `USER CONFIGURATION` 区域，尤其是 `SERIES`、`DATA`、`SUITE`、`EPOCHS`、
   `DEVICE` 和 `PROJECT`。
3. 第一次先设置 `DRY_RUN = True`，点击右上角 Python 三角形，检查模型构建和队列。
4. 确认后设置 `DRY_RUN = False`，再次点击三角形正式训练。
5. 训练期间不要再次点击运行。入口默认使用设备锁，并检查 `nvidia-smi`；检测到同卡已有计算进程时会拒绝启动。

该方式是前台任务：关闭 VS Code、SSH 连接或承载它的终端可能终止训练。这是与 `nohup` 的预期差别。

## 系列与 suite

| `SERIES` | 对应批量脚本 | 可用 `SUITE` |
|---|---|---|
| `SWIFT`（别名 `S`） | `20260824_citrus_swift_batch.py` | `architectures`, `losses`, `all`, `final` |
| `TOPO`（别名 `L`） | `20260824_citrus_topo_batch.py` | `architectures`, `losses`, `all`, `final` |
| `B` | `20260826_citrus_b_batch.py` | `architectures`, `smoke`, `screening`, `losses`, `all`, `final` |
| `C` | `20260828_citrus_c_batch.py` | `smoke`, `controls`, `core`, `architectures`, `losses` |
| `D` | `20260828_citrus_d_batch.py` | `smoke`, `controls`, `core`, `architectures`, `losses` |
| `T` | `20260829_citrus_t_batch.py` | `smoke`, `priority`, `all` |
| `G0830` | `20260830_citrus_g0830_batch.py` | `smoke`, `structure`, `loss`, `all`, `final` |
| `G0839` | `20260830_citrus_g0839_batch.py` | `smoke`, `screen`, `all`, `final` |
| `LIGHT` | `20260830_citrus_light_batch.py` | `smoke`, `screen`, `pareto`, `pr`, `all`, `final` |
| `ORCHID` | `20260901_citrus_orchid_batch.py` | `smoke`, `screen`, `pareto`, `all`, `control`, `final` |
| `SAGE_V2` | `20260902_citrus_sage_batch.py` | `smoke`, `screen`, `all`, `control`, `final`, `aggressive` |
| `SAGE_V3` | `20260902_citrus_sage_v3_batch.py` | `smoke`, `screen`, `all`, `control`, `backbone`, `fusion`, `final` |
| `SAGE_V4` | `20260903_citrus_sage_v4_batch.py` | `smoke`, `screen`, `all`, `control`, `backbone`, `final` |
| `SAGE_V4R` | `20260903_citrus_sage_v4r_batch.py` | `smoke`, `screen`, `structure`, `geometry`, `backbone`, `all`, `control` |

## 资源安全边界

- 每次只允许选择一个 GPU，禁止通过 `0,1` 隐式触发 DDP 多进程。
- `DEVICE_LOCK = True` 使用操作系统咨询锁防止同一用户重复点击；进程被强制结束时锁也会由系统释放。
- `REFUSE_BUSY_GPU = True` 会在训练前检查 GPU 计算进程；发现占用即停止，不会排队或抢占。
- `WORKERS` 只是单个训练的 DataLoader 工作进程数，不代表并行训练模型。共享服务器默认建议保持 `4`；
  SAGE-v4 固定协议会拒绝悄悄降成 `2` 或 `0`。若资源不足，先停止，另建协议后对基线和改进共同修改。
- `SKIP_COMPLETED = True` 只对支持该参数的新系列生效；所有旧脚本仍会拒绝覆盖已有结果目录。
- 不要通过把 `REFUSE_BUSY_GPU` 关闭来绕开其他人的任务；该开关只用于确认 `nvidia-smi` 中显示的是无害驻留进程时的人工审计。

## SAGE-v4 推荐流程

当前重构版为 `SAGE_V4R`（SAGE30 对照和 SAGE40--48），详见
[`docs/SAGE_V4_RECONSTRUCTED_GUIDE.md`](docs/SAGE_V4_RECONSTRUCTED_GUIDE.md)。以下段落描述保留的早期
`SAGE_V4`（SAGE30--35），两者不能混作同一个 suite。

SAGE-v4 修复了 RGB 源图和 `.npy` 缓存被同时计入数据集的问题，因此旧 SAGE-v3 不能直接充当新基线。入口默认
`DRY_RUN = True`；screen 队列应输出五个 `BUILD OK`（all 为六个）。先设 `SUITE="smoke"`、`EPOCHS=3`、独立 `PROJECT` 并改
`DRY_RUN=False` 做短训；随后初筛使用 `EPOCHS = 50`、`SUITE = "screen"` 和新的结果目录，而不是一开始运行 300 轮。
该队列依次运行 SAGE30--34；任何一个失败或用户按 `Ctrl+C` 都不会继续下一模型。每个结果会保存实际加载文件清单、
`loaded_data_summary.json`、官方 `best.pt` 和按 Mask AP50-95 单独选择的 `best_mask.pt`。

所有新 YAML 均可单独使用本地 fork 的标准入口：

```python
from pathlib import Path
from ultralytics import YOLO
from citrus_protocol import fixed_train_args

root = Path(__file__).resolve().parent
model = YOLO(str(root / "0_orange_yaml/SAGE_series/SAGE34_decoupled_geometry.yaml"))
model.load(str(root / "yolo11n-seg.pt"))
model.train(data="/data/sxq/datasets/orange_yolo/data.yaml", epochs=300,
            device="0", seed=42, project="/data/sxq/results/SAGE/SAGE34_single_300EP",
            name="seed42", **fixed_train_args())
```

Windows 单模型训练应把调用放入 `if __name__ == "__main__":` 保护中。不要从另一份 pip 安装的 Ultralytics 导入，
也不要执行升级命令覆盖这个定制 fork。所有新模型不依赖 Mamba/timm 的额外安装。
启动前可检查 `python -c "import ultralytics; print(ultralytics.__file__)"`，路径应指向你刚上传的代码文件夹。
