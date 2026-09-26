# Baseline 第二次预检失败：RF-DETR 撤回版本修复

## 2026-09-25：附件 bug1(2) 的处理

新日志中 MMDetection 已通过预检；现代环境仍安装着 `rfdetr==1.4.0`，整批任务在预检时停止，
尚未开始本次训练。上传代码不等于更新服务器 Python 环境，不能删除版本检查来绕过。

本次只需把本地根目录的 `REPAIR_CITRUS_MODERN_ENV.py` 上传到服务器同名位置，
保留服务器 `RUN_CITRUS_BASELINES_AMP.py` 内自己的 DATA、PROJECT、DEVICE 等设置。
然后逐条执行，上一条成功后再运行下一条：

```bash
cd /data/sxq/code/ultralytics-main-new
/home/amax/sxq/bin/python REPAIR_CITRUS_MODERN_ENV.py --python /home/amax/.conda/envs/citrus_baseline/bin/python3.10
/home/amax/sxq/bin/python RUN_CITRUS_BASELINES_AMP.py --preflight-only
/home/amax/sxq/bin/python RUN_CITRUS_BASELINES_AMP.py
```

修复脚本只安装锁定的 RF-DETR 修订版，使用官方 PyPI、禁用缓存并隔离 pip 用户配置，
不升级依赖。任一步失败立即停止，不启动训练。成功时应先看到 `PREFLIGHT OK` 和
`Modern baseline repair and CPU API preflight verified.`；随后批量程序单独完成 GPU 预检。
若下载或依赖检查失败，请保留完整输出，不要继续开跑。

本次没有修改训练 worker、超参数或批量协议，因此不应仅为此次修复删除结果或更换 PROJECT。
若服务器同时上传了其他改变训练协议的文件，则仍需按批量程序提示使用新的结果目录。

同时为安装脚本补充了现代环境的实际 worker API 预检，避免仅凭 `pip check` 成功就认为安装完成。
本地环境契约回归测试 10 项通过；尚未在用户 Linux 服务器执行包修复或完整训练。

## 根因

服务器已通过 MMDetection 环境预检。现代环境中官方 Ultralytics 也成功从 site-packages 导入，
真正失败项是：环境实际安装 `rfdetr==1.4.0`，代码要求 `1.4.0.post0`。

这不是可以忽略的版本字符串差异。PyPI 将 1.4.0 标记为 yanked，原因是目标检测和实例分割训练
都会在启动后失败；post0 是三小时后发布的未撤回修订包。因此继续使用 1.4.0 可能通过 import，
但不能形成有效 baseline。

## 服务器操作

上传最新 `ultralytics-main-new` 后，在其根目录运行：

```bash
/home/amax/sxq/bin/python REPAIR_CITRUS_MODERN_ENV.py
```

它只修改 `/home/amax/.conda/envs/citrus_baseline` 中的 RF-DETR 包，并依次验证环境身份、强制安装
post0、运行 `pip check`、导入 `RFDETRSegNano`、验证 AMP 配置和官方 Ultralytics 隔离性。

修复完成后先运行：

```bash
/home/amax/sxq/bin/python RUN_CITRUS_BASELINES_AMP.py --preflight-only
```

由于原 `BASELINES_LEGACY78_SCRATCH_300EP` 已保存旧实现 hash，最新代码不得混入该目录。将
`RUN_CITRUS_BASELINES_AMP.py` 现已默认使用全新 AMP 双模式路径；如该路径在服务器已经运行过，再改成另一个全新路径，例如：

```python
"/data/sxq/results/BASELINES/BASELINES_LEGACY78_SCRATCH_FIXED_PAIR_20260923_NEW"
```

另外，服务器 DATA 路径为 `orange_yolo/data.yaml`，但仅凭文件夹名称不能判断数据是否已经更换。
正式论文比较应使用确定的 grouped_dedup 划分，并保持所有基线与改进模型成员一致；
若仅复现旧划分，则不能与 grouped_dedup 结果直接比较。本次环境修复不更改 DATA。

## 保留边界

- 不接受已撤回的 1.4.0；
- 不升级到 1.4.1+，避免改变 baseline 实现；
- 不修改训练超参数或 AMP 设置；
- 不删除失败日志或既有结果；
- 不修改 sxq、citrus_mmdet 或 CUDA 安装。

官方核对：https://pypi.org/pypi/rfdetr/1.4.0/json 与
https://pypi.org/pypi/rfdetr/1.4.0.post0/json。
