# SAGE V5 本地验证记录

日期：2026-09-04。机器：Windows / Ryzen 5 5600H，Python3.9.13，Torch2.8.0+cpu，无CUDA。服务器Python3.8/Torch1.13实机测试尚未执行。

## 已实际执行

- 读取154份历史CSV，140种不同内容；对249个旧YAML建立结构索引。增加新配置后，共256个YAML都能在源码类定义索引中解析模块名，无未解析类名。**这是符号/源码映射，不是256个模型全部训练通过。**
- V4R十组的逐轮结果、参数、实际加载清单、初始化报告、保存的源码摘要核对；检查时源码全部一致。
- 三份已有best_mask.pt重新在本地193张验证图上评估，输出逐GT的面积、solidity、邻接、局部颜色差与匹配记录；不读取test。
- 七个新YAML和两个参照的官方API构建、推理和FLOPs预检全部通过。
- 最终66项测试通过、1项跳过（旧V4R可选fixture路径不存在，不是新模型失败）：新头反向/几何空标签/候选梯度、checkpoint重载、导出模式前向、V4R回归、前台队列/中断/超参锁定等。详见tests_final.log。
- SAGE50–56全部完成真实YOLO.train一轮、验证和best.pt保存。数据是从原train/val分别复制的各4张小型fixture；CPU、256、batch2、workers0、关闭mosaic仅用于管线验证，**不是正式协议或精度证据**。详见smoke_all_new/SMOKE_NOT_FORMAL.json与smoke.log。
- SAGE55通过真实批量入口的一轮测试，保存best_mask、args、初始化报告和清单等回调验证包含在测试中。
- Ruff、git diff空白检查通过，四个新核心入口/模块通过Python3.8语法解析；语法检查不能替代服务器依赖兼容性测试。未运行新模型300轮训练、未验证ONNX/TensorRT、未测真实GPU速度。

## 同环境CPU合成微基准

采用 `cpu_benchmark640_isolated.json`：FP32、batch1、640、2个CPU计算线程、5次预热+20次计时，随机固定模型顺序。计时期间未并行执行本任务的训练/pytest，但仍不是操作系统独占或多次随机区组实验。

| 模型 | 前向中位数/ms | 前向+分割loss+反向/ms |
| --- | ---: | ---: |
| SAGE30 | 118.83 | 455.13 |
| SAGE42 | 128.34 | 477.12 |
| SAGE50 | 108.01 | 427.87 |
| SAGE51 | 118.96 | 476.85 |
| SAGE52 | 102.10 | 437.02 |
| SAGE55 | 110.02 | 486.06 |
| SAGE56 | 102.51 | 435.10 |

本次52相对42前向约快20.4%、反向训练段约快8.4%；对30训练段只快约4.0%，远小于GFLOPs下降比例。55的几何监督仍增加训练耗时，且GFLOPs不会反映训练loss开销。因此不能承诺“服务器提速24%”，也不能拿这张表的前向倒数当部署FPS。

不含数据读取、优化器更新、NMS、掩膜后处理及验证。另存的首次 `cpu_benchmark640.json` 与pytest有并发重叠，明确弃作性能结论，仅保留过程记录；不要挑其中更漂亮的数。

## 文件入口

- `RUN_SAGE_V5.py`：VS Code三角形，默认真实前台训练。
- `20260904_citrus_sage_v5_batch.py`：命令行/启动器共用的串行队列。
- `0_orange_yaml/SAGE_series/SAGE50*.yaml`至`SAGE56*.yaml`：七个新配置。
- `docs/SAGE_V5_EVIDENCE_AND_DESIGN.md`：数据结论、历史取舍、论文及控制理论边界。
- `docs/SAGE_V5_TRAINING.md`：固定超参、运行方式、失败回退。
- `history/history_summary.csv`：可用Excel打开的全结果索引，原始0–1指标值。
- `audit/V4R_RESULTS.md`：十组V4R同协议对照表，百分数。
- `sources/local_source_catalog.json`：59个桌面仓库的目录/remote记录和16份重点源文件指纹；不是“59个仓库逐行全部读完”的宣称。

## 未解决项

新结构能否涨点、tiny AP、颜色扰动鲁棒性、真实split/merge错误和三种子稳定性仍需实验；单轮小fixture零AP不能说明方法无效，也不能说明有效。本fork的PR补零约定保持不变。

最终论文仍需统一测试协议、实例尺度AP、可见凹陷与邻接子集、完整时延和跨模型家族对照；目前仅能交付已通过本地管线检查的候选方法。
