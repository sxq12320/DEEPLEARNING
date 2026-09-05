"""Read-only source provenance and YAML/class mapping; never imports third-party repositories."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/sage_v5_20260904/sources"
DESKTOP = Path("C:/Users/33836/Desktop")


def fingerprint(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    definitions = defaultdict(list)
    for path in (ROOT / "ultralytics/nn/modules").rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        except (SyntaxError, UnicodeError):
            continue
        for item in tree.body:
            if isinstance(item, ast.ClassDef):
                definitions[item.name].append(dict(**fingerprint(path), line=item.lineno))
    mappings, unresolved = [], set()
    for path in (ROOT / "0_orange_yaml").rglob("*.yaml"):
        config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
        if not isinstance(config, dict) or "backbone" not in config:
            continue
        names = sorted({str(row[2]) for row in config.get("backbone", []) + config.get("head", [])})
        refs = {}
        for name in names:
            if name.startswith("nn."):
                refs[name] = "torch.nn standard operator"
            elif name in definitions:
                refs[name] = definitions[name]
            else:
                refs[name] = "UNRESOLVED by top-level class AST scan; may be alias/import"
                unresolved.add(name)
        mappings.append(dict(**fingerprint(path), modules=refs))

    repos = []
    for path in sorted((DESKTOP / "github").iterdir()):
        if not path.is_dir():
            continue
        config = path / ".git/config"
        head = path / ".git/HEAD"
        urls = re.findall(r"url\s*=\s*(.+)", config.read_text()) if config.exists() else []
        revision = None
        if head.exists():
            revision = head.read_text().strip()
            if revision.startswith("ref: "):
                target = path / ".git" / revision[5:]
                revision = target.read_text().strip() if target.exists() else revision
        repos.append(
            dict(
                path=str(path),
                remotes=urls,
                revision=revision,
                scope="directory/remote inventory, not a claim that every source line was read",
            )
        )

    reviewed = [
        "github/PIDNet/models/model_utils.py",
        "github/PIDNet/LICENSE",
        "github/GSCNN/network/gscnn.py",
        "github/FastInst/fastinst/modeling/pixel_decoder/fastinst_encoder.py",
        "github/SparseInst/sparseinst/encoder.py",
        "github/yolact/yolact.py",
        "github/yolact/data/config.py",
        "github/SFM/README.md",
        "github/SFM/SFM_mask2former/mmseg_custom/models/backbones/interp2d.py",
        "github/SCSegamba/README.md",
        "github/SCSegamba/models/GBC.py",
        "Plug-play-modules-main/3. Block（功能模块）/(ECCV 2024) RCM.py",
        "Plug-play-modules-main/3. Block（功能模块）/(CVPR 2024) PKIBlock.py",
        "Plug-play-modules-main/7. Down & Up（池化或上下采样）/(PR2023) HaarDownsampling.py",
        "Plug-play-modules-main/7. Down & Up（池化或上下采样）/(ICCV 2019) CARAFE.py",
        "基于自动控制原理闭环反馈机制的柑橘幼果实例分割网络规划方案_20260902.md",
    ]
    evidence = []
    for relative in reviewed:
        path = DESKTOP / relative
        evidence.append(
            dict(**fingerprint(path), scope="relevant implementation sections inspected; not all code executed")
        )
    payload = dict(
        yaml_modules=mappings,
        unresolved_symbols=sorted(unresolved),
        repository_inventory=repos,
        selected_source_fingerprints=evidence,
        limitations="Current source mapping is not proof of historical runtime source or third-party originality.",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "local_source_catalog.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            dict(
                yamls=len(mappings),
                repositories=len(repos),
                selected_files=len(evidence),
                unresolved=sorted(unresolved),
            )
        )
    )


if __name__ == "__main__":
    main()
