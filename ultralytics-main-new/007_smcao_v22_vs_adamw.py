"""
SMCAO V2.2 vs SMCScheduler V3 vs AdamW — 对比训练脚本
======================================================

在 YOLO11-seg + Apple RGBD 数据集上，对比三种优化器：
1. AdamW (baseline)   — 标准 AdamW + cosine LR
2. SMC V3             — SMCScheduler V3 (滑模面停滞检测 + 有限噪声注入)
3. SMCAO V2.2         — 四项改进：Lévy 抖动 + 负阻尼 + 分数阶记忆 + 自适应 c(f)

使用方式：
    python 007_smcao_v22_vs_adamw.py

输出：
    results/ 目录下三个子目录，分别包含训练结果和曲线
"""

import torch
import random
import numpy as np
import time
import json
from pathlib import Path
from ultralytics import YOLO

SEED = 42
EPOCHS = 40
BATCH = 4
IMGSZ = 400
DATA = r"E:/mastercode/ultralytics-main-new/206_Apple_Amodal.yaml"
PROJECT = r"E:/mastercode/ultralytics-main-new/results"
MODEL_YAML = r"E:/mastercode/ultralytics-main-new/ultralytics/cfg/models/11/yolo11-seg.yaml"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_once(optimizer_name, name_suffix, extra_kwargs=None):
    """运行一次训练，返回结果路径和关键指标"""
    set_seed(SEED)
    yolo = YOLO(MODEL_YAML)
    
    kwargs = dict(
        data=DATA,
        project=PROJECT,
        name=name_suffix,
        optimizer=optimizer_name,
        epochs=EPOCHS,
        patience=50,
        imgsz=IMGSZ,
        batch=BATCH,
        lr0=0.01,
        workers=4,
        device=0,
        cache=False,
        seed=SEED,
        amp=0,
    )
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    
    t0 = time.time()
    results = yolo.train(**kwargs)
    elapsed = time.time() - t0
    
    # 提取关键指标
    metrics = {}
    try:
        metrics["final_fitness"] = results.fitness if hasattr(results, "fitness") else None
        metrics["final_box_map"] = results.box.map if hasattr(results, "box") else None
        metrics["final_seg_map"] = results.seg.map if hasattr(results, "seg") else None
        metrics["final_box_map50"] = results.box.map50 if hasattr(results, "box") else None
        metrics["final_seg_map50"] = results.seg.map50 if hasattr(results, "seg") else None
    except Exception:
        pass
    metrics["time_sec"] = elapsed
    
    return metrics


def main():
    print("=" * 70)
    print("  SMCAO V2.2 vs SMCScheduler V3 vs AdamW — 对比训练")
    print("=" * 70)
    print(f"  Epochs: {EPOCHS}  |  Batch: {BATCH}  |  imgsz: {IMGSZ}")
    print(f"  Data: {DATA}")
    print()
    
    all_results = {}
    
    # ── 1. AdamW baseline ──
    print("\n[1/3] AdamW baseline...")
    adamw_metrics = train_once("AdamW", "02_adamw_baseline")
    all_results["AdamW"] = adamw_metrics
    print(f"  AdamW done: fitness={adamw_metrics.get('final_fitness')}, "
          f"box_map={adamw_metrics.get('final_box_map')}, "
          f"time={adamw_metrics['time_sec']:.1f}s")
    
    # ── 2. SMCScheduler V3 ──
    print("\n[2/3] SMCScheduler V3...")
    smc_metrics = train_once("SMC", "03_smc_v3")
    all_results["SMC_V3"] = smc_metrics
    print(f"  SMC V3 done: fitness={smc_metrics.get('final_fitness')}, "
          f"box_map={smc_metrics.get('final_box_map')}, "
          f"time={smc_metrics['time_sec']:.1f}s")
    
    # ── 3. SMCAO V2.2 ──
    print("\n[3/3] SMCAO V2.2 (Lévy dither + neg damping + fractional memory + adaptive c)...")
    smcao_metrics = train_once("SMCAO", "04_smcao_v22")
    all_results["SMCAO_V22"] = smcao_metrics
    print(f"  SMCAO V2.2 done: fitness={smcao_metrics.get('final_fitness')}, "
          f"box_map={smcao_metrics.get('final_box_map')}, "
          f"time={smcao_metrics['time_sec']:.1f}s")
    
    # ── 汇总输出 ──
    print("\n" + "=" * 70)
    print("  对比结果汇总")
    print("=" * 70)
    print(f"{'Optimizer':<15} {'Fitness':<12} {'Box mAP50-95':<15} {'Seg mAP50-95':<15} {'Box mAP50':<12} {'Time(s)':<10}")
    print("-" * 70)
    for name, m in all_results.items():
        print(f"{name:<15} "
              f"{_fmt(m.get('final_fitness')):<12} "
              f"{_fmt(m.get('final_box_map')):<15} "
              f"{_fmt(m.get('final_seg_map')):<15} "
              f"{_fmt(m.get('final_box_map50')):<12} "
              f"{m.get('time_sec', 0):<10.1f}")
    
    # 保存结果
    save_path = Path(PROJECT) / "smcao_v22_comparison.json"
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n结果已保存至: {save_path}")
    
    # ── 分析 ──
    print("\n" + "=" * 70)
    print("  分析")
    print("=" * 70)
    
    adamw_fitness = all_results.get("AdamW", {}).get("final_fitness", 0) or 0
    smc_fitness = all_results.get("SMC_V3", {}).get("final_fitness", 0) or 0
    smcao_fitness = all_results.get("SMCAO_V22", {}).get("final_fitness", 0) or 0
    
    if adamw_fitness > 0:
        print(f"  SMC V3    vs AdamW: {(smc_fitness/adamw_fitness - 1)*100:+.2f}%")
        print(f"  SMCAO V2.2 vs AdamW: {(smcao_fitness/adamw_fitness - 1)*100:+.2f}%")
        print(f"  SMCAO V2.2 vs SMC V3: {(smcao_fitness/smc_fitness - 1)*100:+.2f}%" if smc_fitness > 0 else "  N/A")
    
    best_name = max(all_results, key=lambda k: all_results[k].get("final_fitness", 0) or 0)
    print(f"\n  最优: {best_name} (fitness={all_results[best_name].get('final_fitness')})")
    
    # SMCAO 特性分析
    print("\n  SMCAO V2.2 四项改进的预期效果：")
    print("    1. Lévy 抖动  → 重尾扰动使参数大跳步，跨越浅层势垒")
    print("    2. 负阻尼     → 梯度消失+高损失时注入能量，冲出局部极小")
    print("    3. 分数阶记忆 → 历史梯度累积推力，飘过平坦区")
    print("    4. 自适应 c   → 高损失强滑模、低损失释放约束、局部极小速度主导")


def _fmt(val):
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.4f}"
    except (TypeError, ValueError):
        return "N/A"


if __name__ == "__main__":
    main()
