# baseline bug1：最小环境修复

日期：2026-09-21。对象：桌面 `bug1` 中的服务器 traceback。

## 结论

程序在第一轮训练开始前就退出了，不是模型结构、显存不足或数据加载错误：

`Runner.from_cfg → _log_env → collect_env → torch.utils.cpp_extension → pkg_resources.packaging`

最后一行是 `ModuleNotFoundError: No module named 'pkg_resources'`。
日志不能确定服务器实际安装了哪个 setuptools 版本，也不能排除其它尚未暴露的环境故障。
PyTorch 2.1.0 使用旧接口；setuptools 70 又移除了该 `packaging` 导出，所以只安装最新版或限定 `<82`
仍可能失败。该隔离环境固定 `setuptools==69.5.1`；不全局降级、不修改 sxq 环境、不跳过环境日志。
官方可复核依据：[setuptools #4376](https://github.com/pypa/setuptools/issues/4376)。

## 服务器立即处理（无需重装 CUDA）

可以先只执行：

```bash
/home/amax/.conda/envs/citrus_mmdet/bin/python -m pip install 'setuptools==69.5.1'
/home/amax/.conda/envs/citrus_mmdet/bin/python -c 'from pkg_resources import packaging; from torch.utils.cpp_extension import CUDA_HOME; from mmengine.utils.dl_utils import collect_env; print(collect_env())'
```

或者上传本次修订的 `code/ultralytics-main-new` 后，在服务器主代码目录执行：

```bash
python REPAIR_CITRUS_MMDET_ENV.py
```

脚本也支持 VS Code 三角形运行；它检查指定目标环境确为 Torch 2.1.0/MMDetection 3.3.0，
只修复 setuptools，然后执行 pip check 和真实 collect_env。当前运行脚本的 Python 不会被安装包。
若环境不是默认位置，使用 `--python /实际环境/bin/python`。`--dry-run` 只展示命令。
使用二进制 MMCV wheel 时，CUDA_HOME=None 并不自动等于故障；不要求为此安装完整 CUDA 编译工具链。

## 本次本地改动

- `baseline_amp/requirements-mmdet.txt`：增加兼容版本锁定。
- `baseline_amp/environment_check.py`：验证旧导出、cpp_extension、collect_env；保留原异常链，不吞掉其它故障。
- `baseline_amp/worker.py`：MMDetection 预检调用真实失败路径，在模型训练前暴露问题。
- `INSTALL_CITRUS_BASELINE_ENVS.py`：安装完成不只 pip check，还执行上述运行时预检。
- `REPAIR_CITRUS_MMDET_ENV.py`：独立前台最小修复入口。
- `baseline_amp/test_environment_check.py`：覆盖两类缺失、正确调用链、其它异常传播、安装器隔离。

所有训练超参数、模型定义、数据集、历史结果均未由此修复改变。

## 重新运行：不要把新协议塞入旧目录

旧报错目录为 `BASELINES_AMP_300EP`；本地主代码已是 `legacy78_scratch_v1_20260920`。
两份可能不同的初始化协议不能靠改 hash 或删 marker 合并。

上传最新代码后，编辑 `RUN_CITRUS_BASELINES_AMP.py` 的 SETTINGS：

```python
"SUITE": "mmdet",
"EPOCHS": 3,
"PROJECT": "/data/sxq/results/BASELINES/MMDET_LEGACY78_FIX_SMOKE_20260921",
"AMP_MODES": [1],  # 延续当前78配方；如做配对审计，设为[1, 0]
```

DATA 保持你已核对过的正式数据实际路径，然后执行：

```bash
python RUN_CITRUS_BASELINES_AMP.py --preflight-only
python RUN_CITRUS_BASELINES_AMP.py
```

3 轮通过后改 EPOCHS=300，并另换全新 PROJECT，再点三角形顺序训练。
已有 YOLO/RF-DETR 结果保留，不需要为了这次 MMDetection 启动失败全部重跑。
但历史结果能否和新模型正式比较，仍取决于划分、初始化和完整训练/评估协议一致性。

新代码会拒绝半成品目录或实现 hash 改变，这是避免覆盖实验，不是 GPU 保护机制。
最直接安全恢复方式是新 PROJECT + SUITE=mmdet；不要删除权重、已完成结果或强改协议记录。

## 验证边界

本地 Windows/CPU：31 项 baseline 测试通过；修复脚本与安装器 dry-run 通过。
未连接你的 Linux GPU 实际执行修复，也没有跑服务器训练。服务器仍需上述预检和 1–3 轮 smoke。
69.5.1 是为隔离旧科研栈采用的兼容约束，不建议用于所有项目，更不是长期安全更新策略。
