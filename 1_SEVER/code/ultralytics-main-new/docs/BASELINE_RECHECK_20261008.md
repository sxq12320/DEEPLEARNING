# Baseline重新检查与修复（2026-10-08）

## 结论

本轮重新检查的是主副本 `baseline_amp/`，入口仍为 `RUN_CITRUS_BASELINES_AMP.py`。
找到了两处可复现的版本/报告问题，并收紧了推理加速的作用范围。
没有证据证明当前共同COCO评估器计算错了AP，也没有把验证改成更低分辨率来制造提速。
旧服务器日志只能证明9月27日的旧版本很慢，不能代表服务器已经加载了10月7日或本轮代码。

| 项目 | 本轮证据与处理 |
|---|---|
| YOLO TXT不是独立快照 | 原代码对图片和TXT均首选硬链接。准备后原地编辑源TXT，会同时改变YOLO训练标签，而已经生成的COCO/RF-DETR标签不变。改为独立TXT快照，并让COCO解析和TXT输出使用同一批字节。 |
| 旧派生标签被修改 | 原先只核对文件大小，同长度改动可能逃过检查。现在TXT逐字节核对，拒绝冲突；发现旧硬链接时要求新PROJECT，不自动解除或覆盖。 |
| 自定义checkpoint冒用best信息 | 旧代码对非best权重也填写best的轮次/EMA身份。现在明确标记selected/custom；custom不填best轮次/角色，必须另建重评目录。selected仍保留原AP重现检查。 |
| 插值加速边界 | 原来安装后只根据no_grad等条件选择native插值；训练也可能包含no_grad辅助计算。现在还必须位于显式推理作用域，包含嵌套、线程隔离和异常清理。这里修的是边界风险，不声称此前训练已被改变。 |

新准备的标签会记录 `independent_txt_snapshot_v1_20261008` 和各TXT的SHA。
图片仍可为硬链接，必须保持不可变；没有把所有图片再复制三遍或增加全数据RAM缓存。
没有修改原图、原JSON、源TXT、数据划分、旧权重或已完成指标。

## 速度问题怎样判定

已有回传旧日志中，193张原图验证约34分钟，单图约10至十几秒，COCO汇总仅数秒。
这支持先查预测、原尺寸掩膜插值/搬运，而不是认定COCO统计是主要瓶颈。
10月7日已加入的native bilinear推理路径、RTMDet/SOLO/YOLO分块掩膜、输入预取、GT/IoU缓存继续保留。
本轮没有把这些已有改动再称作新成果，也不能用CPU测试承诺服务器的CUDA加速倍数。

原图掩膜仍有实际成本。例如100个4096×3072的bool掩膜约1.17GiB，分块只限制临时浮点显存，
并不表示最终CPU掩膜和RLE编码没有代价。24GiB显存容量也不能代表数据搬运快或CPU/RAM不忙。

## 先检查现有权重，不用为排查问题重跑300轮

1. 停止当前属于自己的旧训练进程；不要同时在同一张卡启动测速和训练。
2. 上传更新的整个 `baseline_amp/` 文件夹，以及 `RUN_BASELINE_RECOVERY.py`。
   保留自己在服务器填写的DATA、DEVICE、PYTHONS；不需要重新安装/升级torch、mmcv。
3. 打开 `RUN_BASELINE_RECOVERY.py`，填写：

```python
"RUN_DIR": "/你自己的结果目录/某个具体模型任务",  # 目录内应有job.json
"DEVICE": 1,                                    # 真实物理GPU编号
"MODE": "profile",
"PROFILE_IMAGES": 5,
"CHECKPOINT": "",                               # 默认读取已记录权重，不猜文件
```

`PYTHONS`保留已安装的两个专用环境解释器。选择Python后点击VSCode右上角运行三角形，
或者在主代码目录运行：

```bash
python RUN_BASELINE_RECOVERY.py
```

也可用现有入口的参数，不必修改RUN_DIR：

```bash
python RUN_BASELINE_RECOVERY.py --run-dir /你的具体模型目录 --mode profile --profile-images 5
```

结果写入该模型目录的新 `runtime_profile_时间/output/evaluation_timing.json`，不覆盖旧AP、best或complete。
5张是最大的原图，第一张含热身，不是全验证集平均或部署延迟。profile不计算子集AP。

| 看什么 | 如何解读 |
|---|---|
| predict_seconds大 | 读图等待、GPU前向、掩膜恢复和回传的合计；不能全部叫作网络前向慢 |
| input_wait_seconds大 | 输入准备或存储可能供不上；它已包含在predict时间中，不能再相加 |
| encode_seconds大 | CPU原尺寸掩膜RLE编码耗时明显 |
| native_bilinear_calls | 说明推理插值旁路是否被调用，不是加速倍数；该计数也可能含启动探针 |
| scope/profile_only | 确认测的是原图阶段及纯测速，而不是把quick640与original混比 |

旧legacy测速JSON的逐图记录位于`images`数组；common协议还提供`timing`与`per_image_timing`。
把此JSON和最新日志末尾约80行一起回传，才能判断服务器的当前瓶颈。

## 续训与新实验

未完成MMDetection任务需要续训时，将RECOVERY的MODE改为`resume`。
沿用原优化器、调度、epoch和评价协议，不把旧运行偷偷切换为新common协议；完整权重须包含优化器状态。
正在运行的Python不会热加载你上传的新源码。

新实验仍点击`RUN_CITRUS_BASELINES_AMP.py`。
本轮默认PROJECT改为`BASELINES_CONTROLLED_FASTVAL_20261008`，实现指纹变化后不用旧项目目录绕过检查。
若只跑非YOLO模型，设置`SUITE="non_yolo"`；正式实验前先在新smoke项目跑1–3轮。
不需要nohup、占卡保护或额外人工数据确认。

旧YOLO原生日志与新common COCO结果不能直接混排；各AMP配对的训练设置保持一致。
新数据快照修复只保证随后准备的数据；不能反向宣称已完成的旧实验一定受影响或一定未受影响。

## 没有变化的评价与训练要素

- baseline共同COCO mask AP定义、候选阈值0.001、maxDets=100、固定P/R阈值0.25/IoU0.5。
- quick640每轮监测，原图首轮/每10轮/末轮选模，仅原图Mask AP50–95严格提高才替换best。
- AMP0/AMP1配对、scratch、优化器、学习率、batch、输入尺寸、原生损失与训练预算。
- I/J创新系列EV6Validator；独立旧工作台`4_baseline_choice`及全部已完成结果。

## 本地验证与限制

完整baseline回归：242 passed、22 skipped，16.22秒；Ruff与限定范围git diff检查通过。
另验证了前台队列dry-run和恢复入口帮助；测试不启动真实数据训练。
源码使用原有本地CPU依赖；没有安装或升级服务器环境。
CUDA、真实官方多框架训练和服务器速度仍需上面的现有权重探针验证。
验证记录：`baseline_amp/VERIFICATION_RECHECK_20261008.json`。
