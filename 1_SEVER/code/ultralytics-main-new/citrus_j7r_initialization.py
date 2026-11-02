"""Explicitly paired native weights, without blindly matching shifted layer indices.

This initializes fresh J116--118 graphs only. It is not a historical J90 weight
reproduction or an optimizer/resume operation. Branch parameters are untouched.
"""

import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
CANONICAL_YAML = ROOT / "ultralytics/cfg/models/11/yolo11-seg.yaml"
NATIVE_INITIALIZATION_POLICY = "semantic_yolo11n_seed_v1"


def native_state_sha256(model, layer_map):
    """Hash common weights/buffers by semantic layer, independent of graph indices."""
    digest = hashlib.sha256()
    for semantic, actual in enumerate(layer_map):
        for name, tensor in sorted(model.model[actual].state_dict().items()):
            value = tensor.detach().cpu().contiguous()
            header = ["{:02d}.{}".format(semantic, name), str(value.dtype), list(value.shape)]
            digest.update(json.dumps(header, separators=(",", ":")).encode("utf-8") + b"\0")
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def initialize_native_pair(model, seed):
    """Load a seeded canonical scratch state into explicitly mapped native layers.

    A CPU-only fork protects the caller's Torch RNG. Python, NumPy, CUDA state,
    evidence branch parameters, and optimizer state are not read or modified.
    Every layer class and every state key/shape is checked before any loading.

    Args:
        model: Unfused fresh SegmentationModel with the explicit policy in YAML.
        seed: Nonnegative Python integer for canonical native initialization.

    Returns:
        dict or None: Saved provenance for this fresh initialization, or None if
        the model does not opt into the policy. Do not call this on a resume.
    """
    config = getattr(model, "yaml", None)
    policy = config.get("citrus_native_initialization") if isinstance(config, dict) else None
    if policy is None:
        return None
    if policy != NATIVE_INITIALIZATION_POLICY:
        raise ValueError("Unknown native initialization policy: {}".format(policy))
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Native initialization needs a nonnegative integer seed")
    layer_map = config.get("citrus_native_layer_map")
    if (not isinstance(layer_map, (list, tuple)) or len(layer_map) != 24 or
            any(isinstance(index, bool) or not isinstance(index, int) for index in layer_map) or
            len(set(layer_map)) != 24 or any(index < 0 or index >= len(model.model) for index in layer_map)):
        raise ValueError("citrus_native_layer_map must name 24 distinct, valid model layer indices")

    from ultralytics.nn.tasks import SegmentationModel
    from ultralytics.utils import YAML

    canonical_config = YAML.load(CANONICAL_YAML)
    canonical_config.update(nc=1, scale="n", channels=3)
    # devices=[] is essential: do not enumerate GPUs or initialize CUDA here.
    with torch.random.fork_rng(devices=[]):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        torch.set_rng_state(generator.get_state())
        reference = SegmentationModel(canonical_config, ch=3, nc=1, verbose=False)
    if len(reference.model) != 24:
        raise RuntimeError("Canonical YOLO11-seg changed; review semantic mapping before training")

    states = []
    for semantic, actual in enumerate(layer_map):
        target, source = model.model[actual], reference.model[semantic]
        if type(target) is not type(source):
            raise TypeError("Native semantic layer {} maps to {} ({}) instead of {}".format(
                semantic, actual, type(target).__name__, type(source).__name__))
        target_state, source_state = target.state_dict(), source.state_dict()
        if set(target_state) != set(source_state):
            raise ValueError("Native semantic layer {} has different state keys".format(semantic))
        for name, value in source_state.items():
            if target_state[name].shape != value.shape or target_state[name].dtype != value.dtype:
                raise ValueError("Native semantic layer {} state {} has different shape/dtype".format(semantic, name))
        states.append(source_state)
    # Validation is complete: malformed maps cannot leave a partially loaded model.
    for actual, state in zip(layer_map, states):
        model.model[actual].load_state_dict(state, strict=True)
    record = {
        "policy": policy, "seed": seed, "map": list(layer_map),
        "canonical": {"yaml": CANONICAL_YAML.relative_to(ROOT).as_posix(),
                      "yaml_sha256": hashlib.sha256(CANONICAL_YAML.read_bytes()).hexdigest(),
                      "scale": "n", "nc": 1, "sensor_channels": 3,
                      "weights": "seeded scratch canonical; NOT a historical J90 checkpoint"},
        "common_state_sha256": native_state_sha256(model, layer_map),
        "scope": "mapped native parameters/buffers only; branch, optimizer and external RNG unchanged",
    }
    model.j7r_native_initialization = record
    return record
