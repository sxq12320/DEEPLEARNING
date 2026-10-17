"""YOLO-style Python calls backed by unchanged official baseline implementations.

This is a runtime facade, not a reimplementation in Ultralytics. Models, losses,
optimizers, and prediction decoding remain in their isolated native environments.
Importing this module requires only Python's standard library.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

from .registry import COMMON_VALIDATION, MODELS


def default_pythons():
    """Return the isolated interpreter paths used by the existing installer."""
    if os.name == "nt":
        root = Path.home() / "miniconda3/envs"
        executable = "python.exe"
    else:
        root = Path.home() / ".conda/envs"
        executable = "bin/python"
    return {
        "modern": str(root / "citrus_baseline" / executable),
        "mmdet": str(root / "citrus_mmdet" / executable),
    }


def run_baselines(settings, *, models=None, dry_run=False, preflight_only=False, summarize_only=False):
    """Execute a foreground sequential queue without reading ``sys.argv``.

    Args:
        settings: The SETTINGS dictionary used by RUN_PAPER_BASELINES; include
            RECIPE_POLICY="official_transfer" to select its pretrained recipe.
        models: Optional model IDs to select from the configured suite. Suite
            ordering and paired-AMP ordering are preserved, not redefined.
        dry_run: Print the plan without creating files or loading ML frameworks.
        preflight_only: Run backend checks, without dataset preparation/training.
        summarize_only: Refresh the existing completed-results tables only.

    Returns:
        Resolved project path. In dry-run mode this is a planned path only.
    """
    from .batch import run

    snapshot = copy.deepcopy(settings)
    run(
        snapshot,
        models=models,
        dry_run=dry_run,
        preflight_only=preflight_only,
        summarize_only=summarize_only,
    )
    return Path(snapshot["PROJECT"]).expanduser().resolve()


class OfficialBaseline:
    """One original baseline with a common train/val/profile calling interface.

    Supported names are the exact keys in ``baseline_amp.registry.MODELS``.
    Native checkpoints are not YOLO checkpoints; do not pass them to YOLO().
    """

    def __init__(self, model):
        if model not in MODELS:
            raise ValueError(f"Unknown official baseline: {model!r}. Available: {', '.join(MODELS)}")
        self.model = model

    def train(
        self,
        data,
        project,
        *,
        epochs=300,
        device=1,
        batch=None,
        workers=4,
        seeds=(42,),
        amp_modes=(1, 0),
        pythons=None,
        start_from="",
        skip_runs=(),
        dry_run=False,
        preflight_only=False,
    ):
        """Train with the existing paired-AMP recipe and common validation.

        ``amp_modes=(0,)`` or ``(1,)`` selects one precision mode explicitly;
        the default runs both modes. ``batch`` affects both modes together.
        Model-specific input size, native loss, assignment, and EMA are retained.
        This method does not accept YOLO-only mask/cache/loss overrides.
        """
        settings = dict(
            DATA=str(data),
            PROJECT=str(project),
            DEVICE=device,
            SUITE="all",
            VALIDATION=copy.deepcopy(COMMON_VALIDATION),
            EPOCHS=epochs,
            SEEDS=list(seeds),
            AMP_MODES=list(amp_modes),
            WORKERS=workers,
            BATCHES={} if batch is None else {self.model: batch},
            PYTHONS=default_pythons() if pythons is None else dict(pythons),
            START_FROM=start_from,
            SKIP_RUNS=list(skip_runs),
        )
        return run_baselines(
            settings,
            models=(self.model,),
            dry_run=dry_run,
            preflight_only=preflight_only,
        )

    def val(self, run_dir, *, device=1, checkpoint="", pythons=None, dry_run=False):
        """Re-evaluate an existing job with its saved protocol in a new folder.

        ``run_dir`` is the MODEL directory containing job.json, not the parent
        batch project. No data/resize/metric override is accepted here. Existing
        runs using the older native protocol keep that protocol; there is no
        silent conversion to commonval or EV6.
        """
        return self._evaluate_run(run_dir, "evaluate", device, checkpoint, pythons, dry_run, 0)

    def profile(self, run_dir, *, device=1, checkpoint="", images=5, pythons=None, dry_run=False):
        """Time the largest validation photos without reporting subset AP.

        This uses the same saved decoding/validation policy as the existing
        recovery utility. Timing does not mark a training job complete.
        """
        return self._evaluate_run(run_dir, "profile", device, checkpoint, pythons, dry_run, images)

    def _evaluate_run(self, run_dir, mode, device, checkpoint, pythons, dry_run, images):
        from .recovery import run

        run_dir = Path(run_dir).expanduser().resolve()
        job = json.loads((run_dir / "job.json").read_text(encoding="utf-8"))
        if job["model"] != self.model:
            raise ValueError(f"Run contains {job['model']!r}, not {self.model!r}: {run_dir}")
        settings = dict(
            RUN_DIR=str(run_dir),
            DEVICE=device,
            MODE=mode,
            PROFILE_IMAGES=images,
            CHECKPOINT=str(checkpoint) if checkpoint else "",
            PYTHONS=default_pythons() if pythons is None else dict(pythons),
        )
        return run(settings, dry_run=dry_run)
