# J V3 系列：特权高分辨率老师 × 不完整标注学习

8 个 YAML 的部署图**完全相同**，都是 YOLO11n-seg（与 J20/J00 同图，参数量、FLOPs、延迟一致）。各臂的区别只在训练监督：老师挖出的漏标候选以什么方式进入训练（`pseudo`），以及老师用原图 3×3 fine 视图还是单次 640 推理（`teacher`）。因子见 `citrus_j3_suite.py`，YAML 由 `scripts/generate_citrus_j3_yaml.py` 生成。尚未正式训练，不宣称精度提升。

## 运行

打开项目根目录的 `RUN_CITRUS_J3.py`，设置 `DATA`（**val/test 必须是人工清理后的版本**）、`DEVICE`、`SUITE`、`EPOCHS`、`SEEDS`、`INIT`，然后在 VS Code 右上角点运行，或执行 `python RUN_CITRUS_J3.py`。

- 运行顺序：J50 锚点 → 冻结的 J50 老师（seed 42）只在 train 原图上挖掘漏标候选 → 各学生臂。
- `SUITE="main"` 跑 J50–J55 六个臂；`final` 跑 J50+J53；`oracle` 跑 J59（需要设置 `ORACLE_DATA`，即 train 标签人工清理过的数据）。
- 正式 300 轮使用新的 PROJECT，不覆盖旧结果。

## 直接用 YAML 的注意事项

直接 `YOLO("0_orange_yaml/J_V3_series/J53_privileged_soft.yaml").train(...)` 得到的是普通 YOLO11n-seg 训练：**不会注入老师实例，也不会启用 J3 损失**。J V3 的机制只在 `RUN_CITRUS_J3.py` / `citrus_j3_batch.run` 中生效。

设计、判定规则与证据边界：[设计文档](../../docs/J_V3_DESIGN_20261007.md)。
