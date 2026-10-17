"""Strict queue parameter contracts without importing any training frameworks."""

import copy
from collections import UserDict

import pytest

from baseline_amp.registry import (
    COMMON_VALIDATION,
    CONTROLLED_PROTOCOL_ID,
    MODELS,
    PROTOCOL_ID,
    SUITES,
    YOLO_TRAIN,
    make_queue,
)


def legacy_valid_queue(suite, seeds, epochs, workers, batches=None, amp_modes=(1, 0), validation=None):
    """Frozen previous construction for valid integer/list/tuple inputs."""
    queue = []
    batches = batches or {}
    for seed_index, seed in enumerate(seeds):
        for model_index, model in enumerate(SUITES[suite]):
            order = tuple(amp_modes)
            if (seed_index + model_index) % 2:
                order = order[::-1]
            for amp in order:
                recipe = dict(MODELS[model])
                recipe["batch"] = int(batches.get(model, recipe["batch"]))
                queue.append(dict(
                    model=model,
                    protocol=CONTROLLED_PROTOCOL_ID if validation is not None else PROTOCOL_ID,
                    initialization="scratch",
                    seed=int(seed),
                    amp=bool(amp),
                    epochs=epochs,
                    workers=workers,
                    recipe=recipe,
                    name=f"{model}_amp{amp}_seed{seed}_{epochs}ep",
                ))
                if validation is not None:
                    queue[-1]["validation"] = copy.deepcopy(validation)
    return queue


@pytest.mark.parametrize("suite", list(SUITES))
@pytest.mark.parametrize("amp_modes", [(1, 0), (0, 1), (1,), (0,)])
@pytest.mark.parametrize("validation", [None, COMMON_VALIDATION])
def test_valid_historical_queues_are_unchanged(suite, amp_modes, validation):
    args = dict(suite=suite, seeds=[0, 42, 2**32 - 1], epochs=300, workers=4,
                batches={"mask_rcnn_r50": 1}, amp_modes=amp_modes, validation=validation)
    assert make_queue(**args) == legacy_valid_queue(**args)


@pytest.mark.parametrize("suite", ["unknown", "", None, True, 1, ["all"]])
def test_unknown_suite_is_clear_value_error(suite):
    with pytest.raises(ValueError, match="Invalid suite.*Available"):
        make_queue(suite, [42], 300, 4)


@pytest.mark.parametrize("parameter", ["epochs", "workers"])
@pytest.mark.parametrize("value", [True, False, 1.0, 1.9, "1", None, [], {}])
def test_epochs_and_workers_are_not_coerced(parameter, value):
    args = dict(suite="all", seeds=[42], epochs=300, workers=4)
    args[parameter] = value
    with pytest.raises(ValueError, match=parameter + " must be an integer"):
        make_queue(**args)


@pytest.mark.parametrize("parameter,value", [("epochs", 0), ("epochs", -1), ("workers", -1)])
def test_epochs_and_workers_ranges(parameter, value):
    args = dict(suite="all", seeds=[42], epochs=300, workers=4)
    args[parameter] = value
    with pytest.raises(ValueError, match=parameter + " must be >="):
        make_queue(**args)


@pytest.mark.parametrize("value", [True, False, 42.0, 42.9, "42", None, [], {}])
def test_each_seed_must_be_a_real_integer(value):
    with pytest.raises(ValueError, match="seed must be an integer"):
        make_queue("all", [value], 300, 4)


@pytest.mark.parametrize("seeds", [[], (), "42", b"42", 42, True, None, {42}, {"seed": 42}, iter([42])])
def test_seed_container_must_have_explicit_order(seeds):
    with pytest.raises(ValueError, match="seeds must be a nonempty ordered sequence"):
        make_queue("all", seeds, 300, 4)


@pytest.mark.parametrize("seed", [-1, 2**32, 2**64])
def test_seeds_fit_numpy_and_torch_common_range(seed):
    with pytest.raises(ValueError, match="seed must be 0..4294967295"):
        make_queue("all", [seed], 300, 4)


def test_seed_boundaries_order_and_duplicates():
    jobs = make_queue("rfdetr", (2**32 - 1, 0), 1, 0, amp_modes=(0,))
    assert [job["seed"] for job in jobs] == [2**32 - 1, 0]
    assert all(job["epochs"] == 1 and job["workers"] == 0 for job in jobs)
    with pytest.raises(ValueError, match="Seeds must be unique"):
        make_queue("all", [42, 42], 300, 4)


@pytest.mark.parametrize("batches", [[], (), "", "batch=1", 1, 0, True, False])
def test_batches_require_mapping_even_when_empty_or_falsey(batches):
    with pytest.raises(ValueError, match="batches must be a mapping"):
        make_queue("all", [42], 300, 4, batches=batches)


@pytest.mark.parametrize("value", [True, False, 2.0, 2.9, "2", None, [], {}])
def test_batch_sizes_are_not_truncated_or_coerced(value):
    with pytest.raises(ValueError, match="batch for rtmdet_ins_tiny must be an integer"):
        make_queue("all", [42], 300, 4, batches={"rtmdet_ins_tiny": value})


@pytest.mark.parametrize("value", [0, -1])
def test_batch_sizes_are_positive(value):
    with pytest.raises(ValueError, match="batch for rtmdet_ins_tiny must be >= 1"):
        make_queue("all", [42], 300, 4, batches={"rtmdet_ins_tiny": value})


@pytest.mark.parametrize("key", ["typo", None, 1, ("rtmdet_ins_tiny",)])
def test_batch_override_keys_are_known_model_ids(key):
    with pytest.raises(ValueError, match="Unknown batch override"):
        make_queue("all", [42], 300, 4, batches={key: 2})


def test_other_suite_batch_override_is_still_validated():
    with pytest.raises(ValueError, match="batch for rtmdet_ins_tiny must be an integer"):
        make_queue("yolo", [42], 300, 4, batches={"rtmdet_ins_tiny": 2.9})


@pytest.mark.parametrize("amp_modes", [[], (), "1", 1, True, None, {0, 1}, ["1"], [0.0], [1.0], [2], [None]])
def test_amp_does_not_parse_strings_floats_or_unordered_values(amp_modes):
    with pytest.raises(ValueError, match="AMP_MODES"):
        make_queue("all", [42], 300, 4, amp_modes=amp_modes)


@pytest.mark.parametrize("amp_modes", [[1, 1], [0, False], [True, 1]])
def test_amp_duplicates_rejected_after_boolean_normalization(amp_modes):
    with pytest.raises(ValueError, match="AMP_MODES"):
        make_queue("all", [42], 300, 4, amp_modes=amp_modes)


def test_boolean_amp_is_explicit_and_uses_canonical_integer_job_names():
    assert make_queue("all", [42], 300, 4, amp_modes=(True, False)) == make_queue("all", [42], 300, 4)


def test_mapping_inputs_and_global_model_hyperparameters_are_not_mutated():
    batches = UserDict({"mask_rcnn_r50": 1})
    seeds, amp_modes = [42, 43], [1, 0]
    models_before, train_before = copy.deepcopy(MODELS), copy.deepcopy(YOLO_TRAIN)
    validation = copy.deepcopy(COMMON_VALIDATION)
    jobs = make_queue("all", seeds, 300, 4, batches, amp_modes, validation)
    assert batches == {"mask_rcnn_r50": 1} and seeds == [42, 43] and amp_modes == [1, 0]
    jobs[0]["recipe"]["batch"] = 999
    jobs[0]["validation"]["quick_size"] = 999
    assert MODELS == models_before and YOLO_TRAIN == train_before and validation == COMMON_VALIDATION
