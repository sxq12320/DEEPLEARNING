"""
Optuna 自动化超参数搜索 — SMCScheduler V3 + YOLO11 RGB-D 分割

功能：
1. 定义 Optuna 目标函数，封装 YOLO 模型训练与验证逻辑
2. 搜索空间：学习率 + SMC 核心超参数
3. 优化方向：最大化 mAP50-95(M)（分割 mask mAP）
4. Optuna 中位数剪枝器实现早停
5. 输出最佳超参数组合 + 完整试验日志 CSV

用法：
    python 007_optuna_smc_tune.py --n_trials 30 --epochs 20
    python 007_optuna_smc_tune.py --n_trials 50 --epochs 30 --study_name smc_v3_tune
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# 确保能导入 ultralytics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO

# ═══════════════════════════════════════════════════════════════
# 固定配置
# ═══════════════════════════════════════════════════════════════
DATA_YAML = r"ultralytics-main/206_Apple_Amodal.yaml"
MODEL_YAML = r"ultralytics/cfg/models/11/yolo11-base-rgbd.yaml"
PROJECT_DIR = r"E:/mastercode/ultralytics-main/results"
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Optuna SMC V3 超参数搜索")
    parser.add_argument("--n_trials", type=int, default=30, help="Optuna 试验次数")
    parser.add_argument("--epochs", type=int, default=20, help="每次试验训练 epoch 数")
    parser.add_argument("--study_name", type=str, default="smc_v3_tune", help="Study 名称")
    parser.add_argument("--batch", type=int, default=4, help="批次大小（固定）")
    parser.add_argument("--imgsz", type=int, default=540, help="图像尺寸（固定）")
    parser.add_argument("--workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--device", type=str, default="0", help="训练设备")
    return parser.parse_args()


def create_objective(args):
    """创建 Optuna 目标函数（闭包捕获 args）"""

    def objective(trial: optuna.Trial) -> float:
        # ═══════════════════════════════════════════════════════
        # 1. 定义搜索空间
        # ═══════════════════════════════════════════════════════

        # ── 通用训练超参数 ──
        lr0 = trial.suggest_float("lr0", 1e-4, 1e-2, log=True)

        # ── SMC 核心超参数 ──
        smc_surface_threshold = trial.suggest_float(
            "smc_surface_threshold", 0.01, 0.15, step=0.01
        )
        smc_surface_patience = trial.suggest_int(
            "smc_surface_patience", 30, 200, step=10
        )
        smc_lr_boost = trial.suggest_float(
            "smc_lr_boost", 1.02, 1.2, step=0.02
        )
        smc_noise_scale = trial.suggest_float(
            "smc_noise_scale", 1e-4, 5e-3, log=True
        )
        smc_beta1_low = trial.suggest_float(
            "smc_beta1_low", 0.85, 0.90, step=0.01
        )
        smc_noise_max_steps = trial.suggest_int(
            "smc_noise_max_steps", 3, 20, step=1
        )
        smc_escape_cooldown = trial.suggest_int(
            "smc_escape_cooldown", 50, 200, step=10
        )
        smc_escape_max_duration = trial.suggest_int(
            "smc_escape_max_duration", 10, 40, step=5
        )
        smc_noise_decay = trial.suggest_float(
            "smc_noise_decay", 0.7, 0.95, step=0.05
        )

        # ── 损失增益（微调） ──
        box = trial.suggest_float("box", 5.0, 10.0, step=0.5)
        cls_gain = trial.suggest_float("cls_gain", 0.3, 1.0, step=0.1)
        dfl = trial.suggest_float("dfl", 1.0, 3.0, step=0.5)

        # ═══════════════════════════════════════════════════════
        # 2. 构建 trial 唯一名称
        # ═══════════════════════════════════════════════════════
        trial_name = (
            f"trial_{trial.number:03d}_"
            f"lr{lr0:.1e}_"
            f"sp{smc_surface_patience}_"
            f"lb{smc_lr_boost:.2f}_"
            f"ns{smc_noise_scale:.1e}"
        )

        # ═══════════════════════════════════════════════════════
        # 3. 训练
        # ═══════════════════════════════════════════════════════
        try:
            yolo = YOLO(MODEL_YAML)
            metrics = yolo.train(
                data=DATA_YAML,
                project=PROJECT_DIR,
                name=trial_name,
                optimizer="SMC",
                epochs=args.epochs,
                patience=30,           # trial 级早停 epoch 容忍度
                imgsz=args.imgsz,
                batch=args.batch,
                lr0=lr0,
                workers=args.workers,
                device=args.device,
                cache=False,
                seed=SEED,
                # SMC 超参数
                smc_surface_threshold=smc_surface_threshold,
                smc_surface_patience=smc_surface_patience,
                smc_lr_boost=smc_lr_boost,
                smc_noise_scale=smc_noise_scale,
                smc_beta1_low=smc_beta1_low,
                smc_noise_max_steps=smc_noise_max_steps,
                smc_escape_cooldown=smc_escape_cooldown,
                smc_escape_max_duration=smc_escape_max_duration,
                smc_noise_decay=smc_noise_decay,
                # 损失增益
                box=box,
                cls=cls_gain,
                dfl=dfl,
                # 减少保存开销
                save_period=-1,
                plots=False,
                verbose=False,
            )
        except Exception as e:
            print(f"[Trial {trial.number}] 训练异常: {e}")
            return float("-inf")

        # ═══════════════════════════════════════════════════════
        # 4. 提取指标
        # ═══════════════════════════════════════════════════════
        if metrics is None:
            print(f"[Trial {trial.number}] metrics 为 None")
            return float("-inf")

        results = metrics.results_dict
        # SegmentMetrics: keys 包含 metrics/mAP50-95(B), metrics/mAP50(B),
        #                metrics/mAP50-95(M), metrics/mAP50(M), fitness
        map50_95_mask = results.get("metrics/mAP50-95(M)", 0.0)
        map50_mask = results.get("metrics/mAP50(M)", 0.0)
        fitness = results.get("fitness", 0.0)
        map50_95_box = results.get("metrics/mAP50-95(B)", 0.0)
        map50_box = results.get("metrics/mAP50(B)", 0.0)

        print(
            f"[Trial {trial.number}] "
            f"mAP50-95(M)={map50_95_mask:.4f} "
            f"mAP50(M)={map50_mask:.4f} "
            f"mAP50-95(B)={map50_95_box:.4f} "
            f"fitness={fitness:.4f}"
        )

        # 报告中间值供 pruner 使用（每个 epoch 后 YOLO 已内置验证，
        # 此处用最终值；如需 epoch 级剪枝需回调机制，此处用 trial 级）
        trial.report(map50_95_mask, step=args.epochs)

        # 优化目标：最大化 mask mAP50-95
        return map50_95_mask

    return objective


def run_study(args):
    """运行 Optuna 超参数搜索"""

    # ═══════════════════════════════════════════════════════════
    # 创建 Study
    # ═══════════════════════════════════════════════════════════
    sampler = TPESampler(seed=SEED, n_startup_trials=5)
    pruner = MedianPruner(
        n_startup_trials=5,       # 前 5 次试验不剪枝
        n_warmup_steps=10,        # 至少 10 步不剪枝
        interval_steps=1,         # 每步检查
    )

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",       # 最大化 mAP50-95(M)
        sampler=sampler,
        pruner=pruner,
        storage=None,               # 内存存储；如需持久化可改为 sqlite URL
        load_if_exists=True,
    )

    # ═══════════════════════════════════════════════════════════
    # 执行搜索
    # ═══════════════════════════════════════════════════════════
    objective = create_objective(args)

    print("=" * 70)
    print(f"  Optuna SMC V3 超参数搜索")
    print(f"  Study: {args.study_name}")
    print(f"  Trials: {args.n_trials}")
    print(f"  Epochs/trial: {args.epochs}")
    print(f"  Optimizing: maximize metrics/mAP50-95(M)")
    print(f"  Sampler: TPE (seed={SEED})")
    print(f"  Pruner: MedianPruner")
    print("=" * 70)

    start_time = time.time()
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    elapsed = time.time() - start_time

    # ═══════════════════════════════════════════════════════════
    # 输出结果
    # ═══════════════════════════════════════════════════════════
    best = study.best_trial
    print("\n" + "=" * 70)
    print(f"  搜索完成！总耗时: {elapsed/60:.1f} 分钟")
    print(f"  最佳 Trial: #{best.number}")
    print(f"  最佳 mAP50-95(M): {best.value:.4f}")
    print("-" * 70)
    print("  最佳超参数:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")
    print("=" * 70)

    # ═══════════════════════════════════════════════════════════
    # 保存结果
    # ═══════════════════════════════════════════════════════════
    save_dir = Path(PROJECT_DIR) / args.study_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. 最佳超参数 JSON
    best_params_path = save_dir / "best_params.json"
    best_data = {
        "study_name": args.study_name,
        "best_trial_number": best.number,
        "best_map50_95_mask": best.value,
        "best_params": best.params,
        "n_trials": len(study.trials),
        "total_minutes": round(elapsed / 60, 1),
        "epochs_per_trial": args.epochs,
    }
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(best_data, f, indent=2, ensure_ascii=False)
    print(f"\n[保存] 最佳参数 → {best_params_path}")

    # 2. 完整试验日志 CSV
    csv_path = save_dir / "all_trials.csv"
    # 动态收集所有参数键（不同 trial 可能有不同参数集）
    all_param_keys = set()
    for t in study.trials:
        all_param_keys.update(t.params.keys())
    all_param_keys = sorted(all_param_keys)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["trial_number", "state", "map50_95_mask", "datetime_start", "duration_s"]
        header.extend(all_param_keys)
        writer.writerow(header)
        for t in study.trials:
            row = [
                t.number,
                t.state.name,
                f"{t.value:.6f}" if t.value is not None else "N/A",
                str(t.datetime_start) if t.datetime_start else "",
                f"{t.duration.total_seconds():.1f}" if t.duration else "",
            ]
            for k in all_param_keys:
                row.append(t.params.get(k, "N/A"))
            writer.writerow(row)
    print(f"[保存] 完整试验日志 → {csv_path}")

    # 3. Top-5 试验汇总
    sorted_trials = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value,
        reverse=True,
    )
    print("\n" + "=" * 70)
    print("  Top-5 试验排名:")
    print("-" * 70)
    for i, t in enumerate(sorted_trials[:5]):
        print(f"  #{i+1} Trial {t.number}: mAP50-95(M) = {t.value:.4f}")
        for k, v in sorted(t.params.items()):
            print(f"      {k}: {v}")
    print("=" * 70)

    # 4. 生成可复用的训练脚本代码片段
    best_p = best.params
    reusable_code = f'''
# ═══════════════════════════════════════════════════════════════
# Optuna 最佳超参数自动生成（mAP50-95(M) = {best.value:.4f}）
# ═══════════════════════════════════════════════════════════════
from ultralytics import YOLO

yolo = YOLO(r"{MODEL_YAML}")
yolo.train(
    data=r"{DATA_YAML}",
    project=r"{PROJECT_DIR}",
    name="optuna_best_smc_v3",
    optimizer="SMC",
    epochs=50,                   # 完整训练用更多 epoch
    patience=50,
    imgsz={args.imgsz},
    batch={args.batch},
    lr0={best_p.get('lr0', 0.01)},
    workers={args.workers},
    device="{args.device}",
    seed={SEED},

    # SMC V3 超参数
    smc_surface_threshold={best_p.get('smc_surface_threshold', 0.05)},
    smc_surface_patience={best_p.get('smc_surface_patience', 100)},
    smc_lr_boost={best_p.get('smc_lr_boost', 1.05)},
    smc_noise_scale={best_p.get('smc_noise_scale', 0.001)},
    smc_beta1_low={best_p.get('smc_beta1_low', 0.88)},
    smc_noise_max_steps={best_p.get('smc_noise_max_steps', 10)},
    smc_escape_cooldown={best_p.get('smc_escape_cooldown', 100)},
    smc_escape_max_duration={best_p.get('smc_escape_max_duration', 20)},
    smc_noise_decay={best_p.get('smc_noise_decay', 0.9)},

    # 损失增益
    box={best_p.get('box', 7.5)},
    cls={best_p.get('cls_gain', 0.5)},
    dfl={best_p.get('dfl', 1.5)},
)
'''
    reusable_path = save_dir / "best_train_script.py"
    with open(reusable_path, "w", encoding="utf-8") as f:
        f.write(reusable_code)
    print(f"[保存] 可复用训练脚本 → {reusable_path}")

    return study


if __name__ == "__main__":
    args = parse_args()
    run_study(args)
