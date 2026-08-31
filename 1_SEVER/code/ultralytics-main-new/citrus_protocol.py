"""Single source of truth for formal paper-1 citrus training hyperparameters."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "protocols" / "citrus_paper1_formal_v1.yaml"


def load_protocol() -> dict:
    """Load and minimally validate the formal protocol document."""
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict) or not {"protocol_id", "fixed_train", "phases"}.issubset(protocol):
        raise ValueError(f"Malformed citrus protocol: {PROTOCOL_PATH}")
    return protocol


def protocol_digest(protocol: dict | None = None) -> str:
    """Return a stable SHA-256 identifier for the effective protocol document."""
    payload = protocol or load_protocol()
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def fixed_train_args() -> dict:
    """Return an independent copy of every locked Ultralytics training argument."""
    return deepcopy(load_protocol()["fixed_train"])


def validate_locked_runtime(
    *,
    batch: int,
    imgsz: int,
    workers: int,
    cache: bool | str,
    amp: bool,
    allow_amp_audit: bool = False,
) -> list[str]:
    """Reject silent deviations and return explicitly authorized audit deviations."""
    fixed = fixed_train_args()
    received = {"batch": batch, "imgsz": imgsz, "workers": workers, "cache": cache}
    mismatches = {key: (fixed[key], value) for key, value in received.items() if value != fixed[key]}
    if mismatches:
        raise ValueError(f"Formal protocol is locked; runtime mismatches: {mismatches}")
    if amp != fixed["amp"]:
        if not (allow_amp_audit and amp):
            raise ValueError(f"Formal protocol locks amp={fixed['amp']}; received amp={amp}")
        return ["amp=true (explicit paired audit; not a formal architecture result)"]
    return []


def write_protocol_snapshot(project: Path, additions: dict | None = None) -> tuple[Path, str]:
    """Write the effective protocol and digest once, refusing conflicting snapshots."""
    protocol = load_protocol()
    digest = protocol_digest(protocol)
    if additions:
        protocol["experiment_additions"] = deepcopy(additions)
    directory = project / "_protocol"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "formal_protocol.yaml"
    content = yaml.safe_dump(protocol, allow_unicode=True, sort_keys=False)
    if target.is_file() and target.read_text(encoding="utf-8") != content:
        raise FileExistsError(f"Conflicting protocol snapshot already exists: {target}")
    target.write_text(content, encoding="utf-8")
    (directory / "formal_protocol.sha256").write_text(digest + "\n", encoding="utf-8")
    return target, digest
