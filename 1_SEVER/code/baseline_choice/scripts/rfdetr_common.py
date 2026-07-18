"""Shared helpers for the RF-DETR segmentation baseline."""

from __future__ import annotations

from typing import Any, Dict, Type


def require_rfdetr_class(baseline: Dict[str, Any]) -> Type[Any]:
    """Import the configured RF-DETR class with a version-compatible fallback."""
    try:
        import rfdetr
    except ImportError as exc:
        raise RuntimeError("RF-DETR is unavailable. Activate the citrus_rfdetr environment.") from exc

    class_name = str(baseline["model_class"])
    model_class = getattr(rfdetr, class_name, None)
    if model_class is None and class_name == "RFDETRSegNano":
        model_class = getattr(rfdetr, "RFDETRSegPreview", None)
    if model_class is None:
        raise RuntimeError(f"The installed rfdetr package does not provide {class_name}.")
    return model_class


def model_kwargs(baseline: Dict[str, Any], device: str, weights: str | None = None) -> Dict[str, Any]:
    """Build constructor arguments for the configured segmentation resolution."""
    kwargs: Dict[str, Any] = {
        "device": device,
        "resolution": int(baseline["imgsz"]),
        "positional_encoding_size": int(baseline["positional_encoding_size"]),
    }
    if weights:
        kwargs["pretrain_weights"] = weights
    return kwargs
