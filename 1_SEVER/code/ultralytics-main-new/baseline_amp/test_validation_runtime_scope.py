"""The common validation adapter must opt into, then release, inference-only fixes."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inference_runtime as runtime  # noqa: E402
from validation import validation_state  # noqa: E402


@pytest.mark.parametrize("fail", [False, True])
def test_common_validation_opens_and_cleans_runtime_scope(fail):
    model = torch.nn.Linear(2, 1).train()
    depth = runtime._INFERENCE_SCOPE_DEPTH.get()
    with pytest.raises(RuntimeError, match="prediction failed") if fail else runtime.contextlib.nullcontext():
        with validation_state(model):
            assert runtime._INFERENCE_SCOPE_DEPTH.get() == depth + 1
            assert not torch.is_grad_enabled()
            assert not model.training
            if fail:
                raise RuntimeError("prediction failed")
    assert runtime._INFERENCE_SCOPE_DEPTH.get() == depth
    assert model.training and torch.is_grad_enabled()


def test_common_validation_scope_is_nested_safely():
    model = torch.nn.Linear(2, 1)
    with runtime.inference_runtime_scope():
        depth = runtime._INFERENCE_SCOPE_DEPTH.get()
        with validation_state(model):
            assert runtime._INFERENCE_SCOPE_DEPTH.get() == depth + 1
        assert runtime._INFERENCE_SCOPE_DEPTH.get() == depth
    assert runtime._INFERENCE_SCOPE_DEPTH.get() == 0
