# J 系列：20 组同协议消融候选

这些 YAML 已在项目框架内注册，不需要运行时猴子补丁。未完成正式训练，不宣称精度提升。

## 批量训练

打开项目根目录 `RUN_CITRUS_J.py`，设置自己的 `DATA`、物理 `DEVICE`、`SUITE`、`EPOCHS`。选择环境后点击 VS Code 右上角运行，或 `python RUN_CITRUS_J.py`。默认前台串行，cache=True、AMP=False、scratch。优先 7 组筛选；`SUITE="all"` 共 20 组。正式 300 轮使用新 PROJECT，不覆盖旧结果。

## 单 YAML 入口

```python
from ultralytics import YOLO
from citrus_j_batch import settings

model = YOLO("0_orange_yaml/J_series/J11_synthesis.yaml", task="segment")
model.train(data="/你的数据集/data.yaml", device=0, **settings(300, 42))
```

此示例检验官方 API 可直接使用；它使用标准 trainer，**不自动启用批量脚本的 source-balanced 输入和 EV6 训练器验证口径**，因此指标不能直接放进 J 批量消融表。正式可比单模型实验应在 `RUN_CITRUS_J.py` 设置 `ONLY="J11_synthesis"`。

设计、20 组变量及运行说明：[设计文档](../../docs/J_SERIES_DESIGN_20261001.md)。公式：[数学文档](../../docs/J_SERIES_MATH_20261001.md)。出处与取舍：[文献文档](../../docs/J_SERIES_LITERATURE_20261001.md)。
