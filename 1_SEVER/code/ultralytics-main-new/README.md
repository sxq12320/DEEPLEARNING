# Ultralytics 定制分支 — 未成熟柑橘 RGB 实例分割（活跃主线）

> 更新时间：2026-09-20。本目录是**当前唯一活跃代码库**（服务器 `/data/sxq/` 的回传副本）。
> 根目录 `ultralytics-main-new/` 是 2026-08 的旧开发副本，勿再使用。
> 仓库导航见 `00_START_HERE.md`；最新设计与审查见 `docs/`。

## 当前任务

论文一：**轻量、高精度 RGB 未成熟柑橘实例分割**（单类 `orange_immature`）。
难点：极小果、单图极端尺度跨度、绿果/绿叶混淆、条带状枝叶遮挡造成的深凹可见掩膜、接触实例分离。
范围限定 RGB；不加 RGB-D、amodal、OBB、控制部署或多任务姿态头。

## 系列演进（0_orange_yaml/，423 个 YAML）

- **历史消融库**：B/C/D/F/G/H/L/N/S/SXQ/T、G_0830/0839、Light、ORCHID、SAGE V2–V8、E V1–V8
- **E V9–V12**：持久 P2 细节通路 + 识别/定位任务分流 + 零初始化有界修正。
  V12 内部最优 fine 栅格 mask AP50-95≈78.3%（V12_04）；V12_03 背景误报最少。
  三审一致判定**未达投稿成熟度**（`docs/I_V1_REVIEW_20260920/reviewer{1,2,3}.md`）。
- **I_V1（当前）**：同步双原型掩膜解码——P4 语义原型与已验证细节原型由逐位置门控仲裁，
  全部零初始化、逐参数等价 V12 父模型起步。10 臂含 2 个精确重放锚点。
  设计与停止判据：`docs/I_V1_DESIGN_20260920.md`。

## 训练入口

```bash
pip install -e .                                  # 必须本目录 editable 安装
pytest -q tests/test_citrus_i_v1.py               # 42 项契约测试
python 20260920_citrus_i_v1_batch.py --data /data/sxq/datasets/orange_yolo/data.yaml --suite all --dry-run
```

批量训练用 `RUN_CITRUS_<系列>.py` 前台入口（VS Code ▶），改顶部 DATA/DEVICE/SUITE/EPOCHS；
系列与 suite 清单见 `citrus_foreground.py` 的 `RUNNERS` 和 `FOREGROUND_TRAINING_README.md`。

## 固定协议

正式实验唯一超参来源：`protocols/citrus_paper1_formal_v2_ram.yaml`
（AdamW、lr0=0.001、batch=16、imgsz=640、amp=false、workers=4、cache=ram）。
E/I 系列输入配方：`.5 global/.25 coarse/.25 fine` 源均衡均匀视图；
训练后统一 coarse(0.6)/fine(0.4) 配对栅格评估 + PR 诊断。
数据正式版：grouped_dedup（676/193/96）；V11/V12 服务器 val 成员与之不同，仅内部可比。

## 历史遗留

README 之前的 RGB-D 苹果遮挡内容（SFM/WCAF/DGFFN、`channels: 4`、`206_Apple_Amodal.yaml`）
为休眠支线，模块仍在 `ultralytics/nn/modules/custom_blocks.py` 等文件中注册，勿与主线混淆。
