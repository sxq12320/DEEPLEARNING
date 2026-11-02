# 三个运行入口的验证记录

验证日期：2026-10-25。执行环境为本地Python 3.9.13、PyTorch 2.8.0+cpu。没有执行服务器300轮训练，没有验证服务器CUDA AMP吞吐或最终精度。

## 已通过的验证

1. 40项测试通过：历史78参数核对、队列选择、全图/切片训练分发、续跑保护、完成状态、缓存合并、11个消融配置一致性，以及11个模型的构建、前向、反向、有限梯度与GFLOPs检查。
2. 五个入口/驱动文件通过Python 3.8语法解析。服务器Torch 1.13.1环境仍需运行SMOKE确认，语法检查不等于完整环境兼容验证。
3. 复制两张真实训练图、两张真实验证图，仅在_work工程目录运行。完成全图YOLO11、混合切片YOLO11、预训练V12共三次640输入的一轮训练，以及对应原图/切片评估。
4. 两张训练源图生成26个有效切片，加2个全图视图；两个源图都可切片，没有仅全图回退。实测训练Dataset分别为YOLODataset与MultiScaleViewDataset。
5. V12首个优化步骤前，91.5488%的参数元素与指定预训练文件精确相同。新层与不匹配层允许独立初始化。这验证预训练加载路径，不能证明精度。
6. 对三个已完成运行重新启动入口，best_mask.pt修改时间均保持不变，复用已有训练和评估缓存。
7. V12独立两轮测试：在第一轮checkpoint保存后人为中断，确认checkpoint含优化器状态；重新启动后恢复至第二轮，results.csv恰好为epoch 1、2，随后完成评估。

首次用两张旧图测试时发现所有切片因无效多边形而回退到全图。因此增加training_view_audit.json，并使用可正常裁剪的真实样本重新验证。没有修改原始标签或放宽裁剪规则。全部切片不可用时明确停止，避免无效对照。

## 可追溯产物

- 测试日志：E:/mastercode/_work/20261025_three_runners/tests.log
- 测试结构化结果：E:/mastercode/_work/20261025_three_runners/tests.xml
- 三入口烟雾实验：E:/mastercode/_work/20261025_three_runners/run_20261025_200154/verification.json
- 真正中断续跑实验：E:/mastercode/_work/20261025_three_runners/resume_20261025_200345/verification.json
- 本地复验脚本：E:/mastercode/_work/20261025_three_runners/verify.py、verify_resume.py

本地旧数据仅用于工程检查，不能充当正式数据集实验。服务器上请将三个入口的DATA设为同一份最终数据，并先运行SMOKE，确认环境后关闭SMOKE进行正式训练。
