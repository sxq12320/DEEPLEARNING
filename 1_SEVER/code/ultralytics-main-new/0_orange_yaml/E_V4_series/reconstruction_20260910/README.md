# E V4R：根据 V3 完整结果重构

_2026-09-10；8个因子对照，尚无正式训练精度结论。_

---

入口：主代码目录 `RUN_CITRUS_E_V4.py`，编辑DATA/DEVICE后点VSCode运行；300epoch、cache=True、AMP=False、前台串行。

| 编号 | 配置 |
|---|---|
| V4R00 | 原样E30对照 |
| V4R01 | 仅主干第8层PConv |
| V4R02 | 窄P2细节修正 |
| V4R03 | 有界质量评分 |
| V4R04 | 最深层＋细节 |
| V4R05 | 最深层＋质量 |
| V4R06 | 细节＋质量 |
| V4R07 | 三项组合 |

父目录旧E40–E77未删除，也不是这个新批量入口的默认队列。新YAML无需脚本注入即可使用官方YOLO API；但要复现本系列的源图均衡切片输入，需要用本系列入口的 `SlicedTrainingTrainer`，普通 `model.train(data=原始data)` 不会自动变成切片训练。

完整结果分析、来源、限制、训练说明见主目录 `docs/E_V4_RECONSTRUCTION_20260910.md`。
