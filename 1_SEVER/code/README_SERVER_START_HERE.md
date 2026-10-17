# 服务器上传与运行入口（2026-10-11）

上传本目录后，真正的主项目在 `ultralytics-main-new/`。进入：

```bash
cd /data/sxq/code/ultralytics-main-new
```

外层 `ultralytics-main-new` 是项目，内层 `ultralytics` 是 Python 包，二者不是重复副本，不能删除内层。

## 最新 baseline

唯一批量入口：`RUN_PAPER_BASELINES.py`。
设置真实清洗数据的 DATA、DEVICE 和两个 baseline 环境的 PYTHONS，然后直接运行：

```bash
python RUN_PAPER_BASELINES.py
```

也可在 VSCode 选择 Python 解释器，点击右上角运行三角形。
默认 7 个原版实例分割模型 × AMP=1/0，共 14 项，每项 300 轮，前台顺序运行。
默认新结果目录 `PAPER_BASELINES_TRANSFER_300EP_20261011`，不覆盖之前 50 轮结果。
这是官方预训练配方，不是历史“78版”随机初始化配方。

安装说明和完整运行设置见
[baseline 操作说明](ultralytics-main-new/baseline_amp/README_基线批量运行.md)。
依赖未安装时用 `INSTALL_CITRUS_BASELINE_ENVS.py` 创建隔离环境；已有环境直接复用。
不要在官方 baseline 环境安装改进分支的 `pip install -e .`。

## 改进模型

当前 J4 入口：`RUN_CITRUS_J4.py`；按该文件设置运行参数。
改进模型使用自己的项目环境和 EV6 验证，不把 baseline 的共同 COCO 验证器替换进去。
其余历史模型、YAML、模块及复现工具仍保留。旧系列文档是历史记录，不代表当前默认入口。
外层 `baseline_choice/` 是独有历史工作台，不是新版 baseline 的第二个入口。

## 本目录已瘦身

约 234.22 MiB → 20.45 MiB，主要移出审计图片、重复权重与可再生成缓存。
标签审计页面、CSV 和推理记录完整保存在
`E:/mastercode/1_SEVER/archives/code_cleanup_20261011/LABEL_AUDIT_20261005/`，不需要上传训练服务器。
完整清理与恢复记录在该归档目录的 `README_清理记录.md`。
模型源码、YAML、训练代码、现有结果和数据未删除；训练的 `cache=True` 设置未改变。

本地移动不会自动清理服务器旧副本。上传时只上传当前 code，不要把 archives 一起传上去。
