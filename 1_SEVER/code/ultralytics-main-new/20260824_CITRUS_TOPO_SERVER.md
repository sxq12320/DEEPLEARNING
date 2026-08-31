# CitrusTopo-Seg server training guide

## 1. Environment

Run inside the copied `ultralytics-main-new` directory. This implementation is pure PyTorch and does not require
`mamba-ssm`, `causal-conv1d`, Detectron2, MMDetection, or any custom CUDA extension.

If the server uses CUDA 12.1:

```bash
conda env create -f environment_citrus_topo.yml
conda activate citrus-topo
pip install -e .
```

If the server already has a working PyTorch/CUDA environment, keep it and only run:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
pip install -e .
```

Do not replace a working server PyTorch build merely to match the example CUDA version. `torch.cuda.is_available()`
must print `True` before long training.

## 2. Files that must exist

```text
ultralytics-main-new/
├── 20260824_citrus_topo_batch.py
├── 20260824_citrus_topo_report.py
├── yolo11n-seg.pt
├── 0_orange_yaml/L_series/*.yaml
└── ultralytics/...

/your/data/orange_yolo_grouped_dedup_20260820/
├── data.yaml
├── train/images + train/labels
├── val/images + val/labels
└── test/images + test/labels
```

The copied `data.yaml` may still contain a Windows `E:/...` root. The batch runner creates a portable runtime YAML
with the correct server root; it does not alter the dataset.

## 3. Recommended staged run

First verify the full 16-run queue without training:

```bash
python 20260824_citrus_topo_batch.py \
  --data /your/data/orange_yolo_grouped_dedup_20260820/data.yaml \
  --suite all --dry-run
```

Train the ten structural models sequentially for 300 epochs (same one-Python-runner style as the previous batch):

```bash
python 20260824_citrus_topo_batch.py \
  --data /your/data/orange_yolo_grouped_dedup_20260820/data.yaml \
  --suite architectures --epochs 300 --batch 16 --workers 4 --device 0
```

Generate the first objective table:

```bash
python 20260824_citrus_topo_report.py \
  --project 1_results/L_series/grouped_clean_300ep
```

Then run the six loss ablations on the fixed full architecture:

```bash
python 20260824_citrus_topo_batch.py \
  --data /your/data/orange_yolo_grouped_dedup_20260820/data.yaml \
  --suite losses --epochs 100 --batch 16 --device 0
```

Only after selecting the best screening configuration should the baseline and finalist be trained for 300 epochs.
For the final thesis table, repeat both with seeds 42, 3407, and 20260824 and report mean ± standard deviation.

## 4. Protocol that is locked by the batch runner

- input 640, grouped-dedup train/val/test split;
- COCO-pretrained `yolo11n-seg.pt` initialization;
- AdamW, `lr0=0.001`, `lrf=0.01`, weight decay 0.0005;
- AMP off, seed 42, deterministic mode, mask ratio 4, overlap masks enabled;
- completed run directories are skipped, and every start/completion/failure is appended to `experiment_ledger.jsonl`.

Do not compare these runs with the old leakage-prone split or the accidental AdamW `lr0=0.01` protocol.
