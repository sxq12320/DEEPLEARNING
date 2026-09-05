# SAGE V5 训练与验收

日期：2026-09-04。新模型为SAGE50–56，配置统一放在 `0_orange_yaml/SAGE_series/`。旧版不删除、不重命名、不覆盖。

## 最直接的运行方式

将更新后的 `code/ultralytics-main-new` 同步到服务器相同目录。不能只上传YAML：新模块、modules导出、tasks注册、固定协议和启动脚本必须一起上传。保留服务器自己的数据路径。

在VS Code打开 `RUN_SAGE_V5.py`，选择 `/home/amax/sxq/bin/python` 环境，检查顶部：

```python
DATA = "/data/sxq/datasets/orange_yolo/data.yaml"
DEVICE = "1"  # 你实际独占的那张卡
SUITE = "screen"
EPOCHS = 50
PROJECT = f"/data/sxq/results/SAGE/CITRUS_SAGE_V5_{SUITE.upper()}_{EPOCHS}EP"
DRY_RUN = False
SEEDS = "42"
```

然后点击右上角“运行Python文件”三角形。**默认DRY_RUN=False，会真实训练。** 设置True只是构建检查，打印BUILD OK后回到提示符是正常退出，不是训练已完成。此版项目目录名自动随suite/epochs变化，避免300轮还叫50EP。

当前Python进程前台串行训练，一个模型结束才轮到下一个；不使用nohup、后台shell或多模型并发。DataLoader仍有4个worker，这是读数据，并非同时训练4个模型。顺序按固定种子打乱，屏幕会打印实际队列。

停止：在训练终端按Ctrl+C，队列不会继续启动下一个模型。不要用模糊的pkill python杀掉其他人的任务。右上角强制Stop可能跳过清理；如果GPU被占用保护拒绝启动，先检查自己的旧进程，而不是直接关掉保护。

完整终端输出可直接阅读，不需要tail。每个模型目录仍保存 `results.csv`、`args.yaml`、PR/混淆图、`weights/best.pt`、`weights/best_mask.pt`、`best_mask_selection.json`、`initialization_transfer.json` 和实际加载文件清单。目录存在但未完成会报错保护，不会自动覆盖或偷偷断点续训。

## 队列如何选择

| SUITE | 队列内容 | 建议 |
| --- | --- | --- |
| screen / structure | 30、42、50、51、52 | 首轮用统一50轮筛选；不与历史300轮直接比 |
| geometry | 52、53、54、55 | 结构成立后再做监督2×2消融 |
| backbone | 52、56 | 单独测试G10启发的P5小波主干改动 |
| all | 30、42、50–56共9个 | 功能支持，但暂不建议一次全部300轮 |
| control | 30、42 | 检查基线与旧强参照 |
| smoke | 全队列，限制1–3轮 | 服务器环境换动后先做短训练检查 |

如确定要统一300轮，将EPOCHS改300，保持其他超参不变。最终只对基线、已有强参照和胜出模型做 `SEEDS="42,43,44"`，写入新的输出PROJECT；不要重用已完成实验目录。筛选期间不要用测试集挑选模型。

## 固定超参数

权威配置：`protocols/citrus_paper1_formal_v1.yaml`，本次未修改。启动器会拒绝偷偷更改AMP/batch/imgsz/workers/cache。所有方法的差别只来自明确编号的结构或几何损失因子。

| 项目 | 固定值 |
| --- | --- |
| 数据 | 用户自己的清洗数据YAML；train/val成员须一致，不自动换路径 |
| 输入、批量、读取 | imgsz=640；batch=16；workers=4；cache=False |
| 初始化 | 同一份本地 `yolo11n-seg.pt`，不是某个柑橘旧best.pt |
| 优化器 | AdamW，lr0=.001，lrf=.01，momentum=.937，weight_decay=.0005 |
| warmup | epochs=3，momentum=.8，bias_lr=.1 |
| AMP、dropout | **AMP=False，dropout=0.0** |
| 随机性 | seed=42筛选；最终42/43/44，deterministic=True |
| 主损失 | box=7.5，cls=.5，dfl=1.5，原生实例mask损失保留 |
| 几何损失 | 50/51/52/56均0；53边界.1；54邻接.1；55两项各.1 |
| mask | overlap_mask=True，mask_ratio=4 |
| lr、早停 | cos_lr=False，patience=300，nbs=64，freeze=None |
| 尺度增强 | scale=.5，translate=.1，multi_scale=0，rect=False |
| 颜色增强 | hsv_h=.015，hsv_s=.7，hsv_v=.4，bgr=0 |
| 拼接/混合增强 | mosaic=1，close_mosaic=10；mixup/cutmix/copy_paste=0 |
| 翻转/仿射 | fliplr=.5，flipud=0；degrees/shear/perspective=0 |
| 验证 | split=val，NMS IoU=.7，max_det=300，half=False，plots=True |

新模型共同保留32个mask原型，不通过改小输入图或mask分辨率来“省算力”。无新增Mamba、pytorch-wavelets或CUDA扩展安装要求。现有环境仍必须能够运行这份Ultralytics分割代码；本地测试使用Torch2.8 CPU，服务器Torch1.13仍应做一次短训练验收。

## 单个YAML官方API训练

在这份fork环境中可直接这样使用，不依赖批量脚本做临时注册：

```python
from pathlib import Path
from ultralytics import YOLO
from citrus_protocol import fixed_train_args, load_protocol

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    model = YOLO(str(ROOT / "0_orange_yaml/SAGE_series/SAGE52_dual_route.yaml"))
    model.load(str(ROOT / "yolo11n-seg.pt"))
    settings = fixed_train_args()
    settings.update(load_protocol()["fixed_validation"])
    model.train(
        data="/data/sxq/datasets/orange_yolo/data.yaml",
        epochs=300, seed=42, device=1,
        project="/data/sxq/results/SAGE/CITRUS_SAGE_V5_SINGLE_300EP",
        name="SAGE52_dual_route_seed42", exist_ok=False,
        **settings,
    )
```

单独API默认只保存官方best.pt；如需本套的额外mask最佳权重及完整溯源，直接在 `RUN_SAGE_V5.py` 设置 `ONLY="SAGE52_dual_route"`，其内部仍使用同样的YOLO.train。

## 速度验收：不能只看GFLOPs

批量正式训练前，目标卡空闲时建议运行一次：

```bash
python scripts/benchmark_sage_v5.py --device 1 --batch 16 --imgsz 640 --steps 30 --only SAGE30_official_control,SAGE42_asym_semantic_detail,SAGE52_dual_route --output reports/sage_v5_gpu640_b16.json
```

这是同卡FP32合成输入微基准，包含forward和forward+loss+backward两项，CUDA同步计时；不包括数据读取、优化器更新、NMS、完整验证。若52实测训练步明显慢于42，先返回该JSON排查，不承诺GFLOPs减少就一定按比例加速。若显存不足，不要给某个模型单独偷偷降batch作论文对比；先终止并统一重新定义协议。

正式日志应同时看train/val单轮耗时、GPU利用率和数据等待。nohup本身不会自动争抢GPU，前台运行也不会自动避免共享硬件竞争；本启动器靠单卡锁、占用检查与串行队列降低误并发风险。

## 效果验收与失败回退

首先比较50轮内的30/42/50/51/52，同一数据、初始化、种子与超参。首要Mask AP50–95，同时看Mask AP50、小果Recall/后续COCO AP_small、FP与FN、可见凹陷和邻接子集、Params/GFLOPs与GPU时延。尾20均值只能辅助看后期表现；最终均值±标准差来自独立种子。

如果50明显降精度而51提升，就保留候选细节路由、回退到原型原顺序；如果51误检增加而50不掉点，就优先50的效率方案；如果52成立再测53–55，避免把监督效应误算给结构。56只有胜过52且时延可接受才保留。边界/邻接损失的下降也不能等同于验证精度提高。

重点返回：`results.csv`、`args.yaml`、`best_mask_selection.json`、PR/混淆矩阵、`loaded_data_summary.json`、初始化报告与GPU微基准JSON。新一轮应该用 `best_mask.pt` 做统一子集验证，再决定最终结构，不默认55最强。
