# 交付与验证边界

日期2026-09-06。

- 全历史168份CSV重新读取；154份不同内容。YAML与结果basename候选映射覆盖全部可读结果，不能替代历史源码快照。
- V6仅60/61/63/64确认300轮且有completed标记；62只有118轮。新增V7前，V6保存的源码及初始化checkpoint hash与本地一致。
- 本地SAGE60/64在193张验证图、1049实例完成统一复核；60另完成NMS前后阶段诊断。结果在diagnostics与candidate_stages60.json。未改数据/标签/服务器原始结果。
- 66项相关测试通过，见tests.xml：新模型标准API、前向/反向、空目标、矩形、stride/P2候选数、checkpoint/fuse、GFLOPs、批量回调/串行/中断/完成保护，以及V6回归。
- 四个新模型71–74分别完成1epoch fixture训练/验证/重载，见smoke_all_new。4训练图/4验证图、CPU batch2、256，非精度实验。
- CPU同环境测速见cpu_interleaved640.json（优先）与cpu_benchmark640.json（首轮）。不可当作GPU速度或端到端FPS。
- 标准初始化覆盖记录在initialization。新头预测器独立初始化，不能声称与对照头的预训练覆盖一致。
- 本地Python3.9.13/Torch2.8 CPU；尚未在服务器Python3.8/Torch1.13/CUDA上执行。所有V7正式精度、多种子和困难子集收益待验证。
- 按科研技能要求，将已测结果、解释性假设和创新声明分开；未继续读取控制教材PDF，未把第三方模块目录标注当成经核实的论文成果。
