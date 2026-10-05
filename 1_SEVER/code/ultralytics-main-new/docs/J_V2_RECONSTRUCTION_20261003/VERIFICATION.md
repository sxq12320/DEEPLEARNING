# J_V2工程验证与未解决风险

开始日期2026-10-03；深化检查更新至2026-10-04。
下面的“通过”不是新模型精度、速度或投稿资格的证明。

## 已执行

| 检查 | 实际结果 |
|---|---|
| 全历史结果/当前YAML索引 | 修改架构前311份CSV，去重297条，485份YAML；当前同名映射不是历史Git版本证明 |
| 新结构12臂 | YAML构建、非空/空标注loss、反向、推理输出、native validator契约通过 |
| 导出前向契约 | 普通/共享头均返回预测+proto；没有在本机验证ONNX运行时/TensorRT编译 |
| 部署合并 | 随机冻结BN统计分支等价，整网fuse前后及序列化回读一致，容许浮点误差 |
| 六组回归测试文件 | **172 passed，5 warnings，40.91s**；含历史J、旧J2、EV6、I_V4原生入口 |
| 小夹具前台顺序训练 | J36→J37，各1epoch、4train/2val、128输入、batch2、cache=True，成功完成训练/验证/保存 |
| 选模权重 | 两臂均有best.pt、best_mask.pt、best_mask50.pt、last.pt与协议记录 |
| 数学核验 | SymPy1.14恒等式核验通过，见math_verified.json |
| 静态检查 | 选定新文件Ruff通过；新入口/模块/损失Python3.8 AST语法解析通过 |
| SVG | XML解析及Edge实际渲染、视觉检查通过 |

本机环境：Windows，Python3.9.13，Torch2.8.0+cpu，AMD Ryzen5 5600H。
回归警告包括THOP使用旧distutils比较、当前SciPy对NumPy1.26.4的范围警告。
没有改动用户既有Python环境；本次用了工作区已存在的隔离cpu_deps路径。
Python3.8语法解析**不等于**在服务器Python3.8/Torch1.13/CUDA组合实际测试过。

夹具训练明确标记 `J2_STRUCTURE_SMOKE_NOT_RESEARCH`，
mosaic/copy_paste关闭、warmup0用于检验工程链路；指标为0，不能用于性能比较。
正式运行入口没有这些夹具覆盖。

## 实测CPU不利证据

一次独立测量（没有并发启动本次的训练/测试任务），输入B1×3×640×640、FP32、2线程、9次重复。
中位数仅模型前向，**不含I/O、NMS、mask重建或切片合并**。
没有锁CPU频率/温度，因此不是硬件性能定论或GPU速度预测。

| 模型 | 未融合Params/GFLOPs | 未融合中位ms | 融合中位ms |
|---|---|---:|---:|
| J20 | 2.843M / 10.356 | 117.24 | 99.02 |
| J36 | 1.835M / 9.693 | 165.00 | 142.76 |
| J38 | 1.281M / 7.459 | 139.11 | 158.85 |

**两新图在本次CPU测试都没有比对照快**。低参数目标已在图结构上实现，
低FLOPs也只对计数口径成立；“精度保持、召回改善、GPU更快”仍待真实实验。
融合后J38反而慢也提示测量波动/内存访问不能忽略，不能只摘取最好一次。

THOP遗漏元素运算/部分pool/插值；共享头仍在每个尺度执行，不会因共享自动少三倍MAC。
新P2候选使位置数量增加，与窄proto降低原型生成成本存在张力；训练和验证速度均需单独记录。
**纠正**：标准loss/EV6均插值到input/2，stride4没有降低实例mask矩阵重建面积。
旧stride4原型还绑定四相位；J40/J34同时改变来源与输出分辨率，不能称单项stride归因。

服务器可先做下列有限前向检查，再跑50epoch，避免一上来全套300epoch：

```bash
python scripts/profile_citrus_j2_structure.py \
  --output /data/sxq/results/J/v2/J2_GPU_PROFILE_01 \
  --device cuda:0 --only J20_glance_anchor,J36_structure_complete,J38_compact_complete
```

这里cuda:0是当前进程可见设备编号；如果运行前设置CUDA_VISIBLE_DEVICES，物理编号映射须自行对应。
此脚本的前向时间不替代整个训练每轮、验证每轮或最终部署端到端时间。

## 产物路径

本目录：`history_unique_runs.csv`、`costs_cpu.csv`、`cost_environment.json`、`math_verified.json`。
完整本地日志：

- `E:/mastercode/_work/20261003_j2/regression_final.log`、`regression_final.xml`
- `E:/mastercode/_work/20261003_j2/smoke_verified.log`、`smoke_verified/`
- `E:/mastercode/_work/20261003_j2/profile_isolated.log`、`profile_isolated/`
- `E:/mastercode/_work/20261003_j2/history/`、`evidence/`

`architecture.svg`为旧J36代码对应图，已标记复制限制；`evidence_map.png`为AI辅助的文献取舍概念图，**不是精确计算图或实测结果**。
概念图初版错标5×5/7×7，已重新生成改为3×3/1×1/identity和冻结BN相加，并做视觉检查。
正式网络图以SVG为准，不依据AI图片生成模块连接。

## 尚未完成／不可宣称

- 无新真实数据精度结果、无GPU显存/吞吐测量、无三seed检验、无冻结test的新结论。
- 历史旧验证集与正式grouped_dedup的比较不能当泛化提升。
- 单类混淆矩阵中的背景误报不自动证明是叶片；需要人工复核误报/难例标签。
- Rmax后的零端点不等于某置信度下precision崩溃；AP函数未修改。
- 没有证明闭环控制稳定、CNN/Transformer已完全可解释或网络最优。

下一步先跑默认mechanism六臂，而非24臂全部300轮。若修正版在同协议下掉点或仍慢，明确放弃该实现，
不能为维护创新叙事掩盖不利数据。胜出后再做归因和三seed，而不是继续无证据叠模块。

## 深化修复的实际核验（2026-10-04更新）

| 检查 | 实际结果 |
|---|---|
| J41–J44 | 24 YAML生成一致；16结构臂均构建、非空/空标注前向、loss、反向、推理与EV6契约通过 |
| 解码反例 | 实际旧decoder的四相位差为0；新相位能独立输出一前景三背景 |
| 单项初始化对照 | J36/J41及J38/J44固定seed初始预测和proto在数值容差内相同 |
| RNG/参数 | 转换不额外消耗CPU RNG；末层新增3168参数；相位梯度独立且有限 |
| 插值交换 | float64输出、proto和coeff梯度等价，尚未改正式loss/val实现 |
| PR合成反例 | 实际compute_ap在Rmax补零，即使观察precision全为1也产生零尾；AP代码未改 |
| 八组相关回归测试 | **196 passed，5 warnings，36.02s**，日志regression_pr_final.log/XML |
| 前台夹具 | J41→J42各1epoch、4train/2val、128输入、batch2、cache=True，训练/原生验证/保存完成 |
| 符号推导 | SymPy1.14通过新增面积IoU、相位MAC/参数、噪声反例及网格计数，见math_deep_verified.json |

夹具仍是工程验证而非研究结果，GT仅2个验证实例，AP=0不用于模型优劣判断。
CPU测试不能证明CUDA crop逐像素一致，也没有ONNX/TensorRT实际运行测试。
没有安装或更改用户环境；服务器Python3.8/Torch1.13的实际运行仍待核验。

### 同一次独立CPU重测

没有同时运行本轮训练或测试进程，B1/RGB640/FP32/CPU2线程/9次重复；仅前向，不含后处理。
未锁CPU温度/频率；不能当GPU速度或正式部署结果。

| 模型 | 未融合Params | THOP GFLOPs | 未融合中位ms | 融合中位ms |
|---|---:|---:|---:|---:|
| J20 | 2,842,803 | 10.355763 | 91.23 | 84.83 |
| J36旧图 | 1,834,940 | 9.693030 | 139.68 | 129.69 |
| J41 P3相位修复 | 1,838,108 | 9.691392 | 140.55 | 121.22 |
| J42 P2相位 | 1,837,084 | 9.895782 | 140.71 | 125.39 |
| J43窄P2相位 | 1,283,308 | 7.668582 | 126.31 | 113.70 |

**仍没有CPU加速成功**。不以较低参数/FLOPs掩盖它；更不声称轻量高精度目标已实现。
THOP对nearest/重排的覆盖不同，小差值不能当作相位末层卷积MAC变化；数学上该卷积MAC相同。
建议服务器先测J20/J29/J41/J42/J43的同步前向及完整验证时间；预算不合格者不跑满300epoch。

新增完整日志：`E:/mastercode/_work/20261003_j2_deep/`的
`regression_pr_final.log`/XML、`smoke.log`/`smoke/`、`profile_controlled.log`/`profile_controlled/`。
本目录保存`phase_costs_cpu.csv`、`phase_cost_environment.json`与`math_deep_verified.json`快照。

独立审查报告、逐篇文献章节和反例界限见 [DEEP_REVIEW.md](DEEP_REVIEW.md)。
当前“修复可测试”与“修复有效”仍是两回事。
