"""run_far_batch.py — 无人值守批量训练 73 个柑橘远距离小目标配置。

一句话：把 0_orange_yaml/1_far_small/ 下的全部 yaml 排成队列，逐个调用
train_citrus_seg.py 训练；中断后原地续跑，已完成的不重复。

设计要点
  * 路径全部相对本文件所在目录推导，DATA/PROJECT 直接从 train_citrus_seg.py 的
    常量里读 —— Windows 与服务器同一份代码通用，无需改任何路径。
  * 每个 yaml 的专属训练开关（--freq-loss / --tal-metric ...）从 yaml 头部注释
    的推荐命令里解析，不在本脚本维护第二份真相。
  * 台账 1_batch/batch_ledger.json 记录每个实验的状态/耗时/指标/日志路径；
    默认续跑（跳过 status=done）。
  * 每个实验跑在独立子进程：单个配置炸掉不会带走整批。
  * 优雅停止：在 1_batch/ 下建一个空文件 STOP，当前实验跑完即退出。

常用命令
    python run_far_batch.py --dry-run                     # 只看计划，不训练
    python run_far_batch.py --epochs 50                   # Phase 1 粗筛全量
    python run_far_batch.py --epochs 3 --only F42,F50     # 冒烟指定几个
    python run_far_batch.py --data /data/orange_yolo/data.yaml --epochs 50
    # 断了？重跑同一条命令即可续跑
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
YAML_DIR = HERE / "0_orange_yaml" / "1_far_small"
TRAIN_PY = HERE / "train_citrus_seg.py"
EVAL_PY = HERE / "eval_citrus_seg.py"
STATE_DIR = HERE / "1_batch"
LEDGER = STATE_DIR / "batch_ledger.json"
LOG_DIR = STATE_DIR / "logs"
STOP_FILE = STATE_DIR / "STOP"
SUMMARY_CSV = STATE_DIR / "batch_summary.csv"

# 只允许从 yaml 头部注释里继承这些训练开关（白名单，防注入意外参数）
FLAGS_WITH_VALUE = {
    "--iou-type", "--inner-ratio", "--nwd-ratio", "--tal-metric",
    "--freq-loss", "--optimizer", "--lr0", "--aug-preset",
}
FLAGS_BOOL = {"--slide", "--tal-min-pos"}

# 少数 yaml 把"构成完整方法"的必配开关写在散文注释里、没出现在命令行那一行，
# 这里按各 yaml 原文逐字补齐（yaml 已于 2026-07-26 冻结）。
# 原文标注"可选"的不补（F31 的 --slide、F61 的 --freq-loss）——Phase 1 一次只动一个因子。
HEADER_FLAG_OVERRIDES: dict[str, list[str]] = {
    "F31_yolo11-seg-ours-full":         ["--iou-type", "NWDWise"],
    "F43_yolo11-seg-citrusfar-v2":      ["--iou-type", "NWDWise"],
    "F48_yolo11-seg-citrusformer-net":  ["--iou-type", "NWDWise"],
    "F53_yolo11-seg-citrusformer-plus": ["--iou-type", "NWDWise",
                                         "--tal-metric", "NWD", "--tal-min-pos"],
}

# README §7.2 Phase 1 建议优先级：代价小、预期收益大的先跑
PRIORITY = [
    "F42",   # Shallow-Heavy 算力重分配（反而更轻）
    "F62",   # HSF 高层筛选融合（漏检大头）
    "F22",   # CSFG 跨级引导（P2 头的轻量替代）
    "F50",   # LCE 暗区增强前端
    "F55",   # MWCA 多级小波跨频带注意力
    "F04",   # HWDown 小波下采样
    "F60",   # TGP 纹理先验前端
    "F53",   # CitrusFormer-Plus 精度主打
    "F52",   # Edge-V2 部署线
    "F56",   # 频域全链路
    "SXQNet-seg",       # V1 旗舰
    "SXQNet-V2-nano",   # 端侧
]


def read_const(name: str, default: str = "") -> str:
    """从 train_citrus_seg.py 里读 DATA / PROJECT 常量，避免本脚本重复写死路径。"""
    try:
        text = TRAIN_PY.read_text(encoding="utf-8")
    except OSError:
        return default
    m = re.search(rf'^{name}\s*=\s*r?["\'](.+?)["\']', text, re.M)
    return m.group(1) if m else default


def train_supports_data_flag() -> bool:
    """train_citrus_seg.py 是否已支持 --data（老版本只有写死的 DATA 常量）。"""
    try:
        return '"--data"' in TRAIN_PY.read_text(encoding="utf-8")
    except OSError:
        return False


def parse_header_flags(yaml_path: Path) -> list[str]:
    """从 yaml 头部注释里的推荐训练命令继承专属开关。

    每个 yaml 头部都写了作者建议的完整命令，例如 F56 建议 `--freq-loss 0.1`、
    F53 建议 `--iou-type NWDWise --tal-metric NWD --tal-min-pos`。这里把它当作
    "每个配置该配哪些开关" 的唯一事实来源，避免在本脚本里重抄一份易过期的映射表。
    """
    if yaml_path.stem in HEADER_FLAG_OVERRIDES:
        return list(HEADER_FLAG_OVERRIDES[yaml_path.stem])

    lines: list[str] = []
    with yaml_path.open(encoding="utf-8") as fh:
        for raw in fh:
            if not raw.lstrip().startswith("#"):
                break  # 注释头结束
            lines.append(raw.lstrip().lstrip("#").strip())
    # 反斜杠续行的命令拼回一行
    blob = " ".join(lines).replace("\\ ", " ")
    if "train_citrus_seg.py" not in blob:
        return []
    cmd = blob.split("train_citrus_seg.py", 1)[1]
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        tokens = cmd.split()

    flags: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in FLAGS_BOOL:
            flags.append(tok)
            i += 1
        elif tok in FLAGS_WITH_VALUE and i + 1 < len(tokens):
            flags += [tok, tokens[i + 1]]
            i += 2
        else:
            i += 1  # --model / --name / --pretrained 由本脚本自己给
    return flags


def results_root() -> Path:
    """结果目录：优先从 train_citrus_seg.py 的 PROJECT 常量读，读不到则用 fork 内默认值。"""
    proj = read_const("PROJECT", str(HERE / "1_results" / "ORANGE_WUXI_SEG"))
    return Path(proj)


def discover_yamls() -> list[Path]:
    """F 系列 + SXQNet 家族，按 README 建议优先级排序，其余按文件名。"""
    items = sorted(p for p in YAML_DIR.glob("*.yaml") if p.stem.startswith(("F", "SXQNet")))

    def key(p: Path) -> tuple[int, str]:
        for rank, tag in enumerate(PRIORITY):
            if p.stem == tag or p.stem.startswith(tag + "_"):
                return (rank, p.stem)
        return (len(PRIORITY), p.stem)

    return sorted(items, key=key)


def load_ledger() -> dict:
    if LEDGER.exists():
        return json.loads(LEDGER.read_text(encoding="utf-8"))
    return {}


def save_ledger(ledger: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8")


def build_cmd(yaml_path: Path, epochs: int, batch: int, device: str,
              data_override: str | None) -> list[str]:
    """组装传给 train_citrus_seg.py 的完整命令行。"""
    stem = yaml_path.stem
    name = f"{stem}_{epochs}ep"
    rel_yaml = yaml_path.relative_to(HERE).as_posix()
    args = [
        sys.executable, str(TRAIN_PY),
        "--model", rel_yaml,
        "--pretrained", "yolo11n-seg.pt",
        "--name", name,
        "--epochs", str(epochs),
        "--batch", str(batch),
        "--device", device,
    ]
    if data_override and train_supports_data_flag():
        args += ["--data", data_override]
    # 从 yaml 头部读专属开关（每个配置的唯一事实来源）
    args += parse_header_flags(yaml_path)
    return args


def run_one(yaml_path: Path, epochs: int, batch: int, device: str,
            data_override: str | None, ledger: dict) -> str:
    """训练一个配置；返回 'done' / 'failed'。同时把日志持续写到文件。"""
    stem = yaml_path.stem
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{stem}_{epochs}ep.log"
    cmd = build_cmd(yaml_path, epochs, batch, device, data_override)

    print(f"\n{'='*64}")
    print(f"[BATCH] {stem}  ({epochs} ep)")
    print(f"[CMD]   {' '.join(cmd)}")
    print(f"[LOG]   {log_file}")
    print(f"{'='*64}\n", flush=True)

    ledger[stem] = {
        "status": "running",
        "started": datetime.now().isoformat(timespec="seconds"),
        "epochs": epochs,
        "cmd": cmd,
        "log": str(log_file),
    }
    save_ledger(ledger)

    t0 = time.time()
    ret = None
    with log_file.open("w", encoding="utf-8") as lf:
        lf.write(f"CMD: {' '.join(cmd)}\n{'='*64}\n")
        lf.flush()
        proc = subprocess.Popen(cmd, cwd=str(HERE), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace")
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
            lf.flush()
        ret = proc.wait()

    elapsed = time.time() - t0
    status = "done" if ret == 0 else "failed"
    ledger[stem].update({
        "status": status,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "elapsed_min": round(elapsed / 60, 1),
        "returncode": ret,
    })
    save_ledger(ledger)
    print(f"\n[BATCH] {stem} → {status.upper()}  ({elapsed/60:.1f} min)\n", flush=True)
    return status


def preflight(data_override: str | None, device: str) -> list[str]:
    """起飞前自检：把会让 73 个实验连续失败的问题在第一个实验之前就喊出来。

    返回致命问题清单（非空则不该开跑）。警告类问题只打印不拦。
    """
    fatal: list[str] = []
    print(f"\n{'-'*64}\n起飞前自检\n{'-'*64}")

    # 1) 数据集 yaml —— 最常见的失败原因（DATA 常量指向服务器/旧路径）
    supports_data = train_supports_data_flag()
    if data_override and not supports_data:
        print(f"  [警告] 本机 train_citrus_seg.py 不支持 --data，你给的 --data 会被忽略。")
        print(f"         请改 train_citrus_seg.py 的 DATA 常量，或给它加 --data 参数。")
    data_used = data_override if (data_override and supports_data) else read_const("DATA")
    if not data_used:
        fatal.append("读不到数据集路径：train_citrus_seg.py 里没有 DATA 常量，也没给 --data")
    elif not Path(data_used).exists():
        fatal.append(f"数据集 yaml 不存在：{data_used}\n"
                     f"         → 改 train_citrus_seg.py 第 30 行 DATA 常量指向真实的 data.yaml")
    else:
        print(f"  [OK]   数据集 {data_used}")

    # 2) 预训练权重（协议要求全部实验统一从 COCO 迁移）
    w = HERE / "yolo11n-seg.pt"
    if w.exists():
        print(f"  [OK]   预训练权重 {w.name}")
    else:
        print(f"  [警告] 找不到 {w}，ultralytics 会尝试联网下载；离线服务器请先手动放好。")

    # 3) 环境：魔改版 ultralytics 必须是本 fork（否则自研模块全部 build 失败）
    probe = ("import torch, ultralytics, pathlib;"
             "print('ULTRA', pathlib.Path(ultralytics.__file__).resolve().parent);"
             "print('TORCH', torch.__version__, 'CUDA', torch.cuda.is_available())")
    try:
        r = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                           text=True, timeout=180)
    except (OSError, subprocess.TimeoutExpired) as e:
        r = None
        print(f"  [警告] 环境探测失败：{e}")
    if r is not None and r.returncode != 0:
        fatal.append("当前 python 导入 ultralytics 失败——先在 fork 目录执行 "
                     "`pip install -e .` 安装本 fork（不要 pip install ultralytics 官方版）")
    elif r is not None:
        info = dict(ln.split(" ", 1) for ln in r.stdout.strip().splitlines() if " " in ln)
        ultra_dir = info.get("ULTRA", "?")
        if Path(ultra_dir).parent.resolve() != HERE.resolve():
            print(f"  [警告] 导入的 ultralytics 不在本 fork 内：{ultra_dir}")
            print(f"         自研模块（LCE/TGP/MWCA…）可能不存在 → 大量 yaml 会 build 失败。")
            print(f"         修复：cd {HERE} && pip install -e .")
        else:
            print(f"  [OK]   ultralytics 来自本 fork")
        torch_line = info.get("TORCH", "")
        print(f"  [OK]   torch {torch_line}")
        if device != "cpu" and "CUDA True" not in torch_line:
            fatal.append(f"device={device} 但 torch 检测不到 CUDA。用 --device cpu 或修 GPU 环境。")

    print(f"{'-'*64}")
    return fatal


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="73 个 F 系列/SXQNet 配置的连续批量训练器（可中断、可续跑）。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("--epochs", type=int, default=50,
                   help="每个配置训练轮数（Phase 1 粗筛=50，正式=300）。默认 50")
    p.add_argument("--batch", type=int, default=4, help="batch size，默认 4（协议锁定值）")
    p.add_argument("--device", default="0", help="GPU 编号，默认 0；CPU 用 cpu")
    p.add_argument("--data", default=None,
                   help="数据集 data.yaml 路径。不给则用 train_citrus_seg.py 里的 DATA 常量")
    p.add_argument("--only", default=None,
                   help="只跑匹配的配置，逗号分隔前缀/关键字，如 F42,F62,SXQNet-V1")
    p.add_argument("--skip", default=None, help="跳过匹配的配置，逗号分隔")
    p.add_argument("--priority-only", action="store_true",
                   help="只跑 README 推荐的优先清单（性价比最高的几个）")
    p.add_argument("--limit", type=int, default=None, help="最多跑前 N 个")
    p.add_argument("--retry-failed", action="store_true",
                   help="重跑之前失败的配置（默认只跳过已成功的）")
    p.add_argument("--dry-run", action="store_true", help="只打印计划，不真正训练")
    p.add_argument("--status", action="store_true", help="打印台账汇总后退出")
    return p.parse_args()


def print_status(ledger: dict) -> None:
    if not ledger:
        print("台账为空——还没跑过任何配置。")
        return
    buckets: dict[str, list[str]] = {}
    for stem, rec in ledger.items():
        buckets.setdefault(rec.get("status", "?"), []).append(stem)
    print(f"\n台账：{LEDGER}")
    for st in ("done", "failed", "running"):
        names = sorted(buckets.get(st, []))
        if names:
            print(f"\n[{st.upper()}] {len(names)} 个")
            for n in names:
                rec = ledger[n]
                extra = f"{rec.get('elapsed_min', '?')} min" if st != "running" else "（中断残留）"
                print(f"  - {n:<44} {extra}")
    total_min = sum(r.get("elapsed_min", 0) or 0 for r in ledger.values())
    print(f"\n累计训练时长 ≈ {total_min/60:.1f} 小时")


def main() -> None:
    args = parse_args()
    ledger = load_ledger()

    if args.status:
        print_status(ledger)
        return

    yamls = discover_yamls()
    if not yamls:
        sys.exit(f"没找到任何 yaml：{YAML_DIR}")

    if args.priority_only:
        yamls = [y for y in yamls if any(y.stem.startswith(p) for p in PRIORITY)]
    if args.only:
        keys = [k.strip() for k in args.only.split(",") if k.strip()]
        yamls = [y for y in yamls if any(k in y.stem for k in keys)]
    if args.skip:
        keys = [k.strip() for k in args.skip.split(",") if k.strip()]
        yamls = [y for y in yamls if not any(k in y.stem for k in keys)]

    # 断点续跑：已成功的默认跳过
    pending, skipped = [], []
    for y in yamls:
        rec = ledger.get(y.stem)
        if rec and rec.get("status") == "done":
            skipped.append(y.stem)
        elif rec and rec.get("status") == "failed" and not args.retry_failed:
            skipped.append(y.stem + "(failed,加 --retry-failed 重跑)")
        else:
            pending.append(y)

    if args.limit:
        pending = pending[:args.limit]

    print(f"\n{'='*64}")
    print(f"批量训练计划：{len(pending)} 个待跑 / {len(skipped)} 个跳过")
    print(f"轮数 {args.epochs}ep | batch {args.batch} | device {args.device}")
    print(f"结果目录 {results_root()}")
    print(f"日志目录 {LOG_DIR}")
    print(f"随时中断：Ctrl+C，或建立文件 {STOP_FILE.name}（当前实验跑完即停）")
    print(f"{'='*64}")
    if skipped:
        print(f"\n跳过：{', '.join(skipped[:10])}{' …' if len(skipped) > 10 else ''}")
    print("\n待跑顺序：")
    for i, y in enumerate(pending, 1):
        flags = parse_header_flags(y)
        tail = f"  [{' '.join(flags)}]" if flags else ""
        print(f"  {i:>2}. {y.stem}{tail}")

    if args.dry_run:
        print("\n--dry-run：未实际训练。")
        return
    if not pending:
        print("\n没有待跑配置。")
        return

    if not args.dry_run:
        issues = preflight(args.data, args.device)
        if issues:
            print("\n[FATAL] 以下问题必须先解决才能开始批量训练：")
            for msg in issues:
                print(f"  ✗ {msg}")
            sys.exit(1)

    if STOP_FILE.exists():
        STOP_FILE.unlink()

    n_done = n_fail = 0
    t_start = time.time()
    for i, y in enumerate(pending, 1):
        if STOP_FILE.exists():
            print(f"\n检测到 {STOP_FILE.name}，按要求停止。剩余 {len(pending)-i+1} 个未跑。")
            break
        print(f"\n>>> 进度 {i}/{len(pending)}  已用 {(time.time()-t_start)/60:.0f} min")
        try:
            status = run_one(y, args.epochs, args.batch, args.device, args.data, ledger)
        except KeyboardInterrupt:
            ledger[y.stem]["status"] = "interrupted"
            save_ledger(ledger)
            print(f"\n已中断于 {y.stem}。重跑同一条命令即从这里继续。")
            return
        if status == "done":
            n_done += 1
        else:
            n_fail += 1
            print(f"[BATCH] {y.stem} 失败，继续下一个（日志见 {LOG_DIR}）")

    print(f"\n{'='*64}")
    print(f"批次结束：成功 {n_done} / 失败 {n_fail} / 总耗时 {(time.time()-t_start)/3600:.1f} h")
    print(f"台账 {LEDGER}")
    print(f"查看汇总：python {Path(__file__).name} --status")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
