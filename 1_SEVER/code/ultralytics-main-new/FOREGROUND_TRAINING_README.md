# VS Code 前台串行训练入口

> 更新（2026-09-20）：新系列优先使用各自的专用入口 `RUN_CITRUS_<系列>.py`
>（如 `RUN_CITRUS_I_V1.py`、`RUN_CITRUS_E_V12.py`）——只改文件顶部常量即可，不需要选 SERIES。
>通用入口 `RUN_CITRUS_FOREGROUND.py` 仍然可用，通过 `SERIES` 选择下表中的注册名。

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
| `CITRUS_I_V1`（别名 `I_V1`,`IV1`） | `20260920_citrus_i_v1_batch.py` | `all`, `screen`, `structure`, `smoke`, `priority`, `losses`, `control` |
| `CITRUS_E_V12`（别名 `E_V12`,`EV12`） | `20260918_citrus_e_v12_batch.py` | `all`, `screen`, `structure`, `smoke`, `priority`, `losses`, `control` |
| `CITRUS_E_V11` | `20260917_citrus_e_v11_batch.py` | `all`, `screen`, `structure`, `smoke`, `priority`, `losses`, `control`, `paper` |
| `CITRUS_E_V10` | `20260916_citrus_e_v10_batch.py` | `all`, `screen`, `structure`, `smoke`, `priority`, `losses`, `control` |
| `CITRUS_E_V9` | `20260915_citrus_e_v9_batch.py` | `all`, `screen`, `structure`, `smoke`, `priority`, `losses`, `control` |
| `CITRUS_E_V8` | `20260914_citrus_e_v8_batch.py` | `all`, `screen`, `structure`, `smoke`, `priority`, `combined`, `control` |
| `CITRUS_E_V7` | `20260912_citrus_e_v7_batch.py` | `all`, `screen`, `structure`, `smoke`, `priority`, `combined`, `control` |
| `CITRUS_E_V6` | `20260911_citrus_e_v6_batch.py` | `all`, `screen`, `structure`, `smoke`, `priority`, `combined`, `control` |
| `CITRUS_E_V5` | `20260911_citrus_e_v5_batch.py` | `all`, `screen`, `structure`, `smoke`, `priority`, `combined`, `control` |
| `CITRUS_E_V4R` | `20260910_citrus_e_v4r_batch.py` | `all`, `screen`, `structure`, `smoke`, `priority`, `combined`, `control` |
| `CITRUS_E_V3` | `20260909_citrus_e_v3_batch.py` | `smoke`, `screen`, `structure`, `all`, `control`, `priority`, `combined` |
| `CITRUS_E_V2` | `20260908_citrus_e_v2_batch.py` | `smoke`, `screen`, `structure`, `all`, `control`, `priority`, `guided` |
| `CITRUS_E` | `20260907_citrus_e_batch.py` | `smoke`, `screen`, `structure`, `all`, `control`, `priority`, `guided` |
| `SAGE_V8` | `20260907_citrus_sage_v8_batch.py` | `smoke`, `screen`, `structure`, `all`, `control`, `priority` |
| `SAGE_V7` | `20260906_citrus_sage_v7_batch.py` | `smoke`, `screen`, `structure`, `all`, `control`, `priority`, `refusion`, `refusion_new` |
| `SAGE_V6` | `20260905_citrus_sage_v6_batch.py` | `smoke`, `screen`, `structure`, `geometry`, `backbone`, `all`, `control`, `priority` |
| `SAGE_V5` | `20260904_citrus_sage_v5_batch.py` | `smoke`, `screen`, `structure`, `geometry`, `backbone`, `all`, `control` |
| `SAGE_V4R` | `20260903_citrus_sage_v4r_batch.py` | `smoke`, `screen`, `structure`, `geometry`, `backbone`, `all`, `control` |
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

## 资源安全边界

- 每次只允许选择一个 GPU，禁止通过 `0,1` 隐式触发 DDP 多进程。
- `DEVICE_LOCK = True` 使用操作系统咨询锁防止同一用户重复点击；进程被强制结束时锁也会由系统释放。
- `REFUSE_BUSY_GPU = True` 会在训练前检查 GPU 计算进程；发现占用即停止，不会排队或抢占。
- `WORKERS` 只是单个训练的 DataLoader 工作进程数，不代表并行训练模型。共享服务器默认建议保持 `4`；
  SAGE-v4 固定协议会拒绝悄悄降成 `2` 或 `0`。若资源不足，先停止，另建协议后对基线和改进共同修改。
- `SKIP_COMPLETED = True` 只对支持该参数的新系列生效；所有旧脚本仍会拒绝覆盖已有结果目录。
- 不要通过把 `REFUSE_BUSY_GPU` 关闭来绕开其他人的任务；该开关只用于确认 `nvidia-smi` 中显示的是无害驻留进程时的人工审计。

## 当前推荐流程（I_V1，2026-09-20）

当前主线为 `CITRUS_I_V1`（`RUN_CITRUS_I_V1.py`，设计与停止判据见
[`docs/I_V1_DESIGN_20260920.md`](docs/I_V1_DESIGN_20260920.md)）。先 `DRY_RUN=True` 检查十个臂的
构建与参数/GFLOPs（应输出十个 `BUILD OK`，约 10.6–10.8G）；然后 `SUITE="priority"`、独立 `PROJECT`
跑 I00/I01/I04/I05 四臂筛选。所有新系列训练前统一遵守：不覆盖已有结果目录、`skip_completed`
只对支持的系列生效、完成判定需要 `completed.json`+`results.csv`+`weights/best.pt`。

以下早期段落描述 SAGE_V4R/SAGE_V4，两者不能混作同一个 suite。
历史重构版 `SAGE_V4R`（SAGE30 对照和 SAGE40--48）详见
[`docs/SAGE_V4_RECONSTRUCTED_GUIDE.md`](docs/SAGE_V4_RECONSTRUCTED_GUIDE.md)。

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
