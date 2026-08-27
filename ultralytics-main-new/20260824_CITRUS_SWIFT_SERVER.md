# CitrusSwift-Seg 服务器一键训练说明

## 1. 上传后进入唯一代码根目录

```bash
cd /你的服务器路径/code/ultralytics-main-new
```

目录中再次出现 `ultralytics/` 是正常的：外层 `ultralytics-main-new` 是项目，内层 `ultralytics` 是 Python 包，不是两套代码。

数据集不在 `code` 内。假设服务器数据为：

```text
/data/sxq/datasets/orange_yolo_grouped_dedup_20260820/data.yaml
```

批量程序会读取传入 YAML，并在结果目录生成服务器可用的 runtime YAML，因此本机 `E:/...` 路径不会被带入训练。

## 2. 环境

本轮没有新增依赖，不需要 Mamba、causal-conv1d、Detectron2、MMDetection 或自定义 CUDA 扩展。优先保留服务器已有且能正常使用 CUDA 的 PyTorch 环境：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
pip install -e .
```

`torch.cuda.is_available()` 必须为 `True`。没有可用环境时再执行：

```bash
conda env create -f environment_citrus_topo.yml
conda activate citrus-topo
pip install -e .
```

## 3. 上传后必须先验证

```bash
python -m pytest -q tests/test_citrus_topo.py tests/test_citrus_swift.py
```

预期 27 个测试通过，包括 10 个 YAML 构建/前向、完整损失反向传播、训练专属分支不进入推理、≥90% 预训练继承和 RepContext 融合等价性。

先检查 10 个结构任务，不训练：

```bash
python 20260824_citrus_swift_batch.py \
  --data /data/sxq/datasets/orange_yolo_grouped_dedup_20260820/data.yaml \
  --suite architectures --dry-run
```

## 4. 冒烟测试

先只跑基线和完整候选各 1 epoch：

```bash
python 20260824_citrus_swift_batch.py \
  --data /data/sxq/datasets/orange_yolo_grouped_dedup_20260820/data.yaml \
  --suite architectures --only S00_reference,S08_swift_full \
  --epochs 1 --batch 4 --workers 4 --device 0 \
  --project 1_results/CITRUS_SWIFT_SMOKE
```

显存不足时把 batch 改为 2。冒烟测试只检查程序完整性，不能判断模型好坏。

## 5. 推荐的一键批量流程

先跑 10 个结构的 50 epoch screening：

```bash
nohup python 20260824_citrus_swift_batch.py \
  --data /data/sxq/datasets/orange_yolo_grouped_dedup_20260820/data.yaml \
  --suite architectures --epochs 50 \
  --batch 16 --workers 4 --device 0 \
  --project 1_results/CITRUS_SWIFT_GROUPED_DEDUP_SCREEN50 \
  > citrus_swift_arch_screen50.log 2>&1 &
```

查看进度：

```bash
tail -f citrus_swift_arch_screen50.log
```

生成客观汇总：

```bash
python 20260824_citrus_swift_report.py \
  --project 1_results/CITRUS_SWIFT_GROUPED_DEDUP_SCREEN50
```

之后再跑 10 个损失消融：

```bash
nohup python 20260824_citrus_swift_batch.py \
  --data /data/sxq/datasets/orange_yolo_grouped_dedup_20260820/data.yaml \
  --suite losses --epochs 50 \
  --batch 16 --workers 4 --device 0 \
  --project 1_results/CITRUS_SWIFT_LOSS_SCREEN50 \
  > citrus_swift_loss_screen50.log 2>&1 &
```

只有筛选有效后，才跑正式 300 epoch、3 seeds：

```bash
nohup python 20260824_citrus_swift_batch.py \
  --data /data/sxq/datasets/orange_yolo_grouped_dedup_20260820/data.yaml \
  --suite final --epochs 300 --seeds 42,43,44 \
  --batch 16 --workers 4 --device 0 \
  --project 1_results/CITRUS_SWIFT_FINAL_300EP_3SEEDS \
  > citrus_swift_final_300ep_3seeds.log 2>&1 &
```

如果 screening 后的最终模型不是当前 S08，请先把 runner 中 `final` 的候选改成晋级模型，不能为了省事继续跑已经证明无效的 S08。

## 6. 如果坚持把 10 个结构全部跑 300 epoch

程序仍支持与旧批次完全相同的一键顺序运行方式：

```bash
nohup python 20260824_citrus_swift_batch.py \
  --data /data/sxq/datasets/orange_yolo_grouped_dedup_20260820/data.yaml \
  --suite architectures --epochs 300 \
  --batch 16 --workers 4 --device 0 \
  --project 1_results/CITRUS_SWIFT_ALL_300EP \
  > citrus_swift_all_300ep.log 2>&1 &
```

但这会浪费大量算力，且不符合先 screening 再正式复验的论文实验纪律。

## 7. 速度复测

本机 CPU 数字不能替代服务器 GPU。训练前或训练后在服务器运行：

```bash
python 20260824_citrus_swift_profile.py \
  --device 0 --imgsz 640 --warmup 50 --iterations 200
```

最终部署还要把获胜模型导出 TensorRT FP16，分别测网络前向、NMS、mask decode 和端到端延迟。

## 8. 断点、跳过与实验账本

- 已有 `results.csv` 和 `weights/best.pt` 且 epoch 数足够：自动跳过；
- 目录存在且有 `weights/last.pt`：自动续训；
- 目录存在但没有 `last.pt`：停止并提示，避免覆盖异常结果；
- 单个模型失败：默认记录后继续下一个；增加 `--fail-fast` 可在首个错误停止；
- 每次开始/完成/失败写入 `experiment_ledger.jsonl`，包含模型、数据 YAML、协议、seed、预训练权重与 Git 状态。
