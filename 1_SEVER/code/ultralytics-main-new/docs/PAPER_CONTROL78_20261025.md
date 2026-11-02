# 78版训练参数：两个模型、前台顺序运行

## 入口

打开根目录 RUN_PAPER_CONTROL78.py，修改 DATA、DEVICE、PROJECT，然后在VS Code选择服务器Python解释器，点击“运行Python文件”。不要用Interactive Window复用已经初始化CUDA的进程。

也可在该代码目录运行：

~~~bash
python -u RUN_PAPER_CONTROL78.py
~~~

默认依次训练C00_yolo11n_78_seed42（T00原生YOLO11n YAML）和C01_v12_04_78_seed42（现有V12_04 YAML）。
没有新增网络、修改旧YAML或覆盖原结果。默认300轮预算、patience=100，继承原78版早停语义，可能提前正常结束；不是强制跑满300轮。

## 必须改的路径

~~~python
DATA = "/你实际存放的数据集/data.yaml"
DEVICE = "1"
PROJECT = "/你存放结果的位置/PAPER_CONTROL78_20261025"
~~~

默认DATA指向正式分组去重数据。请填你实际确认过的路径；不强制核对划分清单，不要求confirm-data，不自动切换回旧orange_yolo，也不自动改标签。
GPU编号为单张物理卡。没有“GPU被占用就禁止运行”的保护机制；只禁止两个进程同时写同一PROJECT，防止结果损坏。建议每卡一个训练任务。

## 已冻结的参数

来源：001_3_yolo11-seg_adamw/args.yaml，历史Mask AP50=78.635%。

| 设置 | 本次两个模型共同值 |
|---|---|
| epochs / patience | 300 / 100 |
| batch / imgsz / workers | 16 / 640 / 4 |
| optimizer / lr0 / lrf | AdamW / 0.001 / 0.01 |
| weight_decay / momentum | 0.0005 / 0.937 |
| warmup_epochs / warmup_momentum / warmup_bias_lr | 3 / 0.8 / 0.1 |
| AMP / cache | True / True |
| dropout / mask_ratio | 0.1 / 4 |
| mosaic / close_mosaic | 1.0 / 10 |
| copy_paste / mixup / cutmix | 0 / 0 / 0 |
| cos_lr / deterministic / seed | False / True / 42 |
| 初始化 | 两者从YAML随机初始化，不加载预训练权重 |
| 训练输入 | 原数据全图＋原版增强，不生成混合切片训练集 |

其余HSV、几何增强等显式写在citrus_control78_batch.py的RECIPE78。未复用后期fixed_train_args，以免继承AMP=False、mask_ratio=2或copy_paste=0.3。
旧args中的pretrained=True配合YAML入口不证明已加载权重；已有legacy78审计判定该路线为scratch。本次明确pretrained=False实现scratch，不假装与旧配置文本逐字相同，也不保证换数据后仍恰好78分。

与上一版计划相比，按本次用户要求采用全图训练，不启用计划中提到的共同混合视图训练。若以后要做混合视图，需另建双方一致的对照，不混入本次。

## 验证口径

- 训练mask_ratio=4，两个模型训练中都用现有EV6Validator的input/2真值验证；这是统一口径，不声称逐位复现历史78运行。
- 每个模型按验证Mask AP50–95保存best_mask.pt，best.pt仍由原框架保留。
- 训练后best_mask.pt自动执行全图、外框切片合并、掩膜切片选择、边界感知切片选择四种val诊断；核心对比是全图与边界感知十视图。
- 使用现有citrus_paper_eval缓存候选，coco_secondary=False，不运行MMDet或baseline COCO验证器，也不自动查看test。
- 缓存身份检查仅防止重启时使用不属于当前权重/数据的旧预测，不要求数据集带划分指纹或人工确认。
- V12_04自带质量、Tiny、边界及邻实例等监督仍存在；本轮比较完整方法，不可把全部差异单独归因于主干。

## 正式跑之前

先将SMOKE=True运行一次：只训练1轮，后处理只取2幅val图片；输出自动加_SMOKE后缀。检查结束后改回False正式运行。CPU smoke无法验证GPU AMP稳定性。
DRY_RUN=True仅打印队列及参数，不导入Torch、不检查远端数据、不训练、不写结果。只看队列不能替代smoke。

## 停止和继续

Ctrl+C停止后不会启动下一个模型。再运行同一个入口：

- 未训练完且有有效last.pt：恢复优化器/EMA继续，并恢复已有best_mask记录。
- 训练已正常结束：不重训，只补未完成的验证。
- 验证中断：复用已完成图片候选缓存。
- 全部完成：跳过训练并重用有效验证报告。
- 训练参数、YAML或本入口实现改变：使用新PROJECT，拒绝混合旧实验。
- 有CSV但缺少可恢复checkpoint：明确报错，不自动从头覆盖。

改SEEDS=[42,43,44]会复用完成的42并增加后两个seed；不需要把第一轮重跑。可用ONLY="candidate"只继续候选，但不要改变共同配方。

## 查看结果

PROJECT/CONTROL78_RESULTS.csv：模型×推理方式汇总，列名明确percent。
各模型目录含args.yaml、control78_protocol.json、effective_runtime.json、results.csv、
best_mask_selection.json、weights/best_mask.pt、training_finished.json、
eval_val/paired_metrics.json和eval_val/manual_review.csv。
manual_review.csv是待人工复核预测，不是人工确认标签。

前台直接显示进度，不需要nohup。本地检查结论以测试记录为准；没有服务器GPU不能承诺AMP兼容或训练精度。本次不更改服务器环境。

## 已完成验证（2026-10-25）

- 16项针对性测试通过：历史参数核对、队列、输出互斥、早停、恢复保护、最佳权重保护、普通全图数据路由及mask_ratio=4前向/反向等。
- Python 3.8语法检查通过；不依赖argparse.BooleanOptionalAction。
- 两模型在复制的2幅train＋2幅val图上，完成640输入、batch2、workers0、各1轮CPU训练和四臂配对验证。
- 重复运行完整入口成功复用训练/验证结果，best_mask.pt未被改写。
- 在第1轮保存后主动中断，再恢复至第2轮成功；CSV保留两轮，优化器/EMA恢复路径生效。
- 修复旧eval_citrus_sliced.predict_view对零实例掩膜插值报错：合法空预测直接返回空候选，不改有预测时的结果或AP规则。
- 本地Torch2.8.0+cpu，无GPU；没有运行300轮，没有验证服务器Torch1.13/CUDA AMP的数值稳定性或精度。
- 本地用户site中的Matplotlib与NumPy版本不匹配，验证进程选择了已安装的兼容Matplotlib3.5.2，不安装/修改任何全局包；生产入口会在训练前检查绘图依赖可导入。

原始检查产物：E:/mastercode/_work/20261025_control78/。
