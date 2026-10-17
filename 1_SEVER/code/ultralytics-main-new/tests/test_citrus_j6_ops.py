"""Synthetic engineering checks, not Citrus AP or CUDA timing measurements."""

import pytest
import torch
from torch.nn import functional as F

from ultralytics.nn.modules.citrus_j6_ops import (
    IndependentDiscovery,
    InstanceResidual,
    ObservableRelation,
    SharedRGBPrefix,
    bounded_pair_edges,
    bounded_set_selection,
    exact_reference_selection,
    masked_relation_loss,
    normalized_roi_grid,
    observable_pair_features,
    packed_roi_sample,
    project_roi_logits,
    rival_roi_sample,
    roi_logits_on_grid,
    set_energy,
    sparse_observable_pair_features,
    sparse_relation_matrix,
)


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(16102026)
    yield
    torch.set_num_threads(previous)


def test_prefix_and_discovery_shapes_backward():
    rgb = torch.rand(2, 3, 65, 79, requires_grad=True)
    detail, scout = SharedRGBPrefix()(rgb)
    assert detail.shape == (2, 8, 33, 40)
    assert scout.shape == (2, 16, 17, 20)
    maps = IndependentDiscovery()(scout)
    assert [x.shape[1] for x in maps.values()] == [1, 2, 2]
    sum(x.square().mean() for x in maps.values()).backward()
    assert torch.isfinite(rgb.grad).all() and rgb.grad.abs().sum() > 0


def test_packed_roi_matches_direct_value_gradient():
    feature = torch.randn(2, 3, 13, 17, requires_grad=True)
    boxes = torch.tensor([[[.1, .2, .5, .6], [.2, .1, .9, .8]], [[0., 0., 1., 1.], [.3, .4, .7, .9]]])
    grid = 2 * normalized_roi_grid(boxes, side=5) - 1
    direct = torch.stack([torch.stack([F.grid_sample(feature[b:b + 1], grid[b, n:n + 1],
                                                   align_corners=False, padding_mode="border")[0]
                                      for n in range(2)]) for b in range(2)])
    actual = packed_roi_sample(feature, boxes, side=5)
    assert torch.allclose(actual, direct, atol=2e-6)
    weights = torch.randn_like(actual)
    first = torch.autograd.grad((actual * weights).sum(), feature, retain_graph=True)[0]
    second = torch.autograd.grad((direct * weights).sum(), feature)[0]
    assert torch.allclose(first, second, atol=2e-6)


def test_roi_analytic_ramp_box_gradient_and_empty():
    xx, yy = (torch.arange(5) + .5) / 5, (torch.arange(3) + .5) / 3
    image = (xx[None] + 2 * yy[:, None])[None, None]
    boxes = torch.tensor([[[.1, .2, .9, .8]]], requires_grad=True)
    actual = packed_roi_sample(image, boxes, side=3)
    axis = (torch.arange(3) + .5) / 3
    expected = (.1 + .8 * axis)[None] + 2 * (.2 + .6 * axis)[:, None]
    assert torch.allclose(actual[0, 0, 0], expected, atol=2e-7)
    actual.sum().backward()
    assert torch.isfinite(boxes.grad).all() and boxes.grad.abs().sum() > 0
    empty = packed_roi_sample(image, torch.empty(1, 0, 4), side=4)
    assert empty.shape == (1, 0, 1, 4, 4)


@pytest.mark.parametrize("box", [[0., 0., 0., 1.], [0., 0., 1.1, 1.], [0., 0., float("nan"), 1.]])
def test_roi_invalid_boxes(box):
    with pytest.raises(ValueError):
        packed_roi_sample(torch.rand(1, 3, 8, 8), torch.tensor([[box]]))


def test_common_projection_support_and_outside():
    logits = torch.ones(1, 2, 1, 4, 4, requires_grad=True)
    boxes = torch.tensor([[[0., 0., .5, .5], [.5, .5, 1., 1.]]])
    projected, support = project_roi_logits(logits, boxes, (8, 10))
    assert projected.shape == support.shape == (1, 2, 1, 8, 10)
    assert support[0, 0, :, :4, :5].all()
    assert not support[0, 0, :, 4:].any()
    assert torch.equal(projected[~support], torch.full_like(projected[~support], -20.))
    projected[support].sum().backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        roi_logits_on_grid(logits, boxes, torch.full((1, 2, 2), float("nan")))


def test_neighbor_sampling_uses_target_world_coordinates():
    logits = torch.stack((torch.full((1, 4, 4), 3.), torch.full((1, 4, 4), 7.)))[None]
    boxes = torch.tensor([[[0., 0., .5, .5], [.25, 0., .75, .5]]])
    neighbors = torch.tensor([[[1, -1], [0, 1]]])
    values, support = rival_roi_sample(logits, boxes, neighbors, side=4)
    assert values.shape == (1, 2, 2, 1, 4, 4)
    assert support[0, 0, 0, :, :, 2:].all()
    assert not support[0, 0, 0, :, :, :2].any()
    assert torch.equal(values[0, 0, 0][support[0, 0, 0]], torch.full((8,), 7.))
    assert not support[:, :, 1].any()  # absent and self neighbor are unknown


def residual_inputs(count=3):
    return [torch.randn(1, count, 8, 5, 5, requires_grad=True),
            torch.randn(1, count, 8, 5, 5, requires_grad=True),
            torch.randn(1, count, 32, requires_grad=True),
            torch.randn(1, count, 1, 5, 5, requires_grad=True),
            torch.randn(1, count, 1, 5, 5, requires_grad=True)]


def test_residual_initial_identity_bound_gradients_and_permutation():
    model, inputs = InstanceResidual(), residual_inputs()
    assert torch.equal(model(*inputs), inputs[3])
    nn = torch.nn
    nn.init.normal_(model.output.weight, std=.1)
    output = model(*inputs)
    assert (output - inputs[3]).abs().max() <= 2.000001
    output.square().mean().backward()
    for item in inputs:
        assert torch.isfinite(item.grad).all() and item.grad.abs().sum() > 0
    order = torch.tensor([2, 0, 1])
    assert torch.allclose(model(*(item[:, order] for item in inputs)), output[:, order], atol=1e-6)


def test_zero_init_residual_can_learn_and_empty():
    model, inputs = InstanceResidual(), residual_inputs(2)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    model(*inputs).square().mean().backward()
    assert model.output.weight.grad.abs().sum() > 0
    assert model.input.weight.grad.abs().sum() == 0
    optimizer.step()
    optimizer.zero_grad()
    model(*inputs).square().mean().backward()
    assert model.input.weight.grad.abs().sum() > 0
    empty = model(*residual_inputs(0))
    assert empty.shape == (1, 0, 1, 5, 5)


def test_sparse_dense_common_evidence_value_parity():
    masks, supports = torch.rand(2, 5, 60) > .5, torch.rand(2, 5, 60) > .4
    boxes = torch.tensor([[[.1, .1, .6, .6]] * 5] * 2)
    edges = bounded_pair_edges(boxes, max_neighbors=4)["edges"]
    sparse = sparse_observable_pair_features(masks, supports, edges)
    dense = observable_pair_features(masks, supports)
    b = torch.arange(2)[:, None]
    left, right = edges.unbind(-1)
    assert torch.allclose(sparse["features"], dense["features"][b, left, right], atol=1e-7)
    assert torch.equal(sparse["known"], dense["known"][b, left, right])


def test_sparse_unknown_common_domain_and_foreground():
    masks = torch.ones(1, 2, 12, dtype=torch.bool)
    supports = torch.zeros_like(masks)
    supports[0, 0, :5], supports[0, 1, 7:] = True, True
    edges = torch.tensor([[[0, 1], [-1, -1]]])
    result = sparse_observable_pair_features(masks, supports, edges)
    assert not result["known"].any()
    assert result["features"][0, 0, 0] == 0
    result = sparse_observable_pair_features(torch.zeros_like(masks), torch.ones_like(supports), edges)
    assert not result["known"].any()
    assert torch.equal(result["features"][0, 1], torch.zeros(5))


def test_sparse_ignore_unobserved_and_literal_set_oracle():
    masks, supports = torch.rand(1, 4, 25) > .4, torch.rand(1, 4, 25) > .4
    edges = torch.tensor([[[0, 1], [0, 2], [1, 3]]])
    first = sparse_observable_pair_features(masks, supports, edges, min_common=1, min_foreground=1)
    changed = sparse_observable_pair_features(masks ^ ~supports, supports, edges, min_common=1, min_foreground=1)
    assert torch.equal(first["features"], changed["features"])
    for e, (i, j) in enumerate(edges[0].tolist()):
        common = {p for p in range(25) if supports[0, i, p] and supports[0, j, p]}
        left = {p for p in common if masks[0, i, p]}
        right = {p for p in common if masks[0, j, p]}
        assert first["common_samples"][0, e] == len(common)
        assert first["common_foreground_union"][0, e] == len(left | right)
        assert first["features"][0, e, 0] == pytest.approx(len(left & right) / max(len(left | right), 1))


def test_edge_cap_validity_determinism_and_canonical_ties():
    boxes = torch.tensor([[[.1, .1, .8, .8]] * 10])
    result = bounded_pair_edges(boxes, max_neighbors=1, max_edges=3)
    assert result["counts"].item() == 3
    assert result["edges"].tolist() == [[[0, 1], [0, 2], [0, 3]]]
    assert torch.equal(result["edges"], bounded_pair_edges(boxes, 1, 3)["edges"])
    valid = torch.tensor([[True] * 3 + [False] * 7])
    result = bounded_pair_edges(boxes, max_neighbors=4, max_edges=10, valid=valid)
    assert result["counts"].item() == 3
    assert (result["edges"][result["valid"]] < 3).all()
    assert (result["edges"][~result["valid"]] == -1).all()


def test_sparse_optional_identity_cues_symmetric_and_differentiable():
    masks, supports = torch.ones(1, 2, 16, dtype=torch.bool), torch.ones(1, 2, 16, dtype=torch.bool)
    edges = torch.tensor([[[0, 1]]])
    query = torch.randn(1, 2, 4, requires_grad=True)
    boxes = torch.tensor([[[.1, .2, .5, .7], [.2, .1, .8, .8]]], requires_grad=True)
    scores = torch.tensor([[.3, .7]], requires_grad=True)
    first = sparse_observable_pair_features(masks, supports, edges, queries=query, boxes=boxes, scores=scores)
    changed = sparse_observable_pair_features(masks.flip(1), supports.flip(1), edges, queries=query.flip(1),
                                             boxes=boxes.flip(1), scores=scores.flip(1))
    assert first["features"].shape[-1] == 20
    assert torch.allclose(first["features"], changed["features"], atol=1e-7)
    model = ObservableRelation(appearance_channels=4, include_geometry=True, include_scores=True)
    relation = model(first)
    masked_relation_loss(relation, torch.ones(1, 1, dtype=torch.bool), torch.ones(1, 1, dtype=torch.bool)).backward()
    for item in (query, boxes, scores):
        assert torch.isfinite(item.grad).all() and item.grad.abs().sum() > 0


def test_relation_deployment_gate_persists_and_scatter():
    evidence = {"features": torch.ones(1, 1, 5), "known": torch.ones(1, 1, dtype=torch.bool),
                "edges": torch.tensor([[[0, 1]]])}
    model = ObservableRelation()
    relation = model(evidence)
    assert relation["known"].all() and not relation["selection_known"].any()
    _, known = sparse_relation_matrix(relation, 2)
    assert not known.any()
    model.mark_trained()
    relation = model(evidence)
    values, known = sparse_relation_matrix(relation, 2)
    assert known.sum() == 2 and values[0, 0, 1] == values[0, 1, 0]
    copied = ObservableRelation()
    copied.load_state_dict(model.state_dict())
    assert copied.relation_trained.item()
    duplicate = {"logits": torch.zeros(1, 2), "edges": torch.tensor([[[0, 1], [0, 1]]]),
                 "selection_known": torch.ones(1, 2, dtype=torch.bool)}
    with pytest.raises(ValueError):
        sparse_relation_matrix(duplicate, 2)


def test_relation_unknown_nan_and_exact_loss_gradient():
    logits = torch.tensor([[[float("nan"), 2., -1.], [2., float("nan"), float("nan")],
                            [-1., float("nan"), float("nan")]]], requires_grad=True)
    known = torch.tensor([[[0, 1, 1], [1, 0, 0], [1, 0, 0]]], dtype=torch.bool)
    labels = torch.tensor([[[0, 1, 0], [1, 0, 0], [0, 0, 0]]], dtype=torch.bool)
    loss = masked_relation_loss({"logits": logits, "known": known}, labels, known)
    assert torch.allclose(loss, (F.softplus(torch.tensor(-2.)) + F.softplus(torch.tensor(-1.))) / 2)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0, 1] == pytest.approx(float((torch.sigmoid(torch.tensor(2.)) - 1) / 2))
    empty = torch.full((1, 2), float("nan"), requires_grad=True)
    loss = masked_relation_loss({"logits": empty, "known": torch.zeros(1, 2, dtype=torch.bool)},
                                torch.zeros(1, 2, dtype=torch.bool), torch.zeros(1, 2, dtype=torch.bool))
    loss.backward()
    assert loss.item() == 0 and torch.equal(empty.grad, torch.zeros_like(empty))
    with pytest.raises(ValueError):
        masked_relation_loss({"logits": empty, "known": torch.ones(1, 2, dtype=torch.bool)},
                             torch.zeros(1, 2, dtype=torch.bool), torch.ones(1, 2, dtype=torch.bool))


def test_deployment_selection_untrained_relation_cannot_suppress():
    unary, logits = torch.tensor([.8, .7, -.1]), torch.full((3, 3), 100.)
    known = ~torch.eye(3, dtype=torch.bool)
    result = bounded_set_selection(unary, logits, known, budget=3)
    assert result["selected"].tolist() == [True, True, False]
    assert result["energy"].item() == pytest.approx(1.5)
    trained = bounded_set_selection(unary, logits, known, budget=3, relation_trained=True)
    assert trained["selected"].tolist() == [True, False, False]


def test_selection_star_exposes_local_reference_gap():
    unary = torch.tensor([1.01, 1., 1., 1., 1., 1., 1.])
    logits = torch.full((7, 7), -20.)
    logits[0, 1:] = logits[1:, 0] = 20.
    known = ~torch.eye(7, dtype=torch.bool)
    result = bounded_set_selection(unary, logits, known, budget=6, relation_trained=True, improvement_rounds=8)
    exact, best = exact_reference_selection(unary, logits, known, budget=6)
    assert result["greedy_energy"].item() == pytest.approx(1.01)
    # Strict one-swap improvement cannot cross this local barrier. Retain this
    # negative control instead of claiming that a local optimizer is exact.
    assert result["selected"].tolist() == [True, False, False, False, False, False, False]
    assert exact.tolist() == [False, True, True, True, True, True, True]
    assert best - result["energy"].item() == pytest.approx(4.99)


def test_selection_budget_zero_empty_ties_and_negative():
    for unary in (torch.tensor([]), torch.tensor([1., 1.]), torch.tensor([-1., -2.])):
        n = unary.numel()
        logits, known = torch.zeros(n, n), torch.zeros(n, n, dtype=torch.bool)
        assert not bounded_set_selection(unary, logits, known, 0)["selected"].any()
    result = bounded_set_selection(torch.ones(2), torch.zeros(2, 2), torch.zeros(2, 2, dtype=torch.bool), 1)
    assert result["indices"].tolist() == [0]
    assert not bounded_set_selection(torch.tensor([-1., -2.]), torch.zeros(2, 2),
                                      torch.zeros(2, 2, dtype=torch.bool), 2)["selected"].any()


def test_selection_finite_step_budget_and_reference_gap():
    for seed in range(8):
        generator = torch.Generator().manual_seed(seed)
        unary = torch.rand(6, generator=generator) - .15
        raw = torch.randn(6, 6, generator=generator)
        logits = (raw + raw.T) / 2
        known = ~torch.eye(6, dtype=torch.bool)
        result = bounded_set_selection(unary, logits, known, 3, relation_trained=True,
                                       max_steps=10, improvement_rounds=4)
        _, optimal = exact_reference_selection(unary, logits, known, 3)
        assert result["selected"].sum() <= 3 and result["steps"] <= 10
        assert result["energy"] >= result["greedy_energy"] - 1e-6
        assert optimal >= result["energy"].item() - 1e-6
        expected = set_energy(unary, logits, known, result["selected"])
        assert torch.allclose(expected, result["energy"], atol=1e-6)


@pytest.mark.parametrize("which", ["unary", "known_pair", "asymmetry", "cap", "budget"])
def test_selection_rejects_invalid(which):
    unary, pair, known, budget = torch.ones(3), torch.zeros(3, 3), ~torch.eye(3, dtype=torch.bool), 2
    if which == "unary":
        unary[0] = float("nan")
    elif which == "known_pair":
        pair[0, 1] = pair[1, 0] = float("nan")
    elif which == "asymmetry":
        known[0, 1] = False
    elif which == "cap":
        unary, pair, known = torch.ones(129), torch.zeros(129, 129), torch.zeros(129, 129, dtype=torch.bool)
    else:
        budget = -1
    with pytest.raises(ValueError):
        bounded_set_selection(unary, pair, known, budget)


def test_unknown_nan_pairs_ignored_and_energy_gradient():
    unary = torch.tensor([1., -2.], requires_grad=True)
    logits = torch.full((2, 2), float("nan"), requires_grad=True)
    known = torch.zeros(2, 2, dtype=torch.bool)
    output = set_energy(unary, logits, known, torch.tensor([True, False]))
    output.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))
    assert bounded_set_selection(unary.detach(), logits.detach(), known, 2)["indices"].tolist() == [0]


def test_sparse_empty_and_caps_and_nonfinite():
    masks = torch.zeros(1, 0, 12, dtype=torch.bool)
    edges = bounded_pair_edges(torch.empty(1, 0, 4))["edges"]
    result = sparse_observable_pair_features(masks, masks, edges)
    assert result["features"].shape == (1, 0, 5)
    result = sparse_observable_pair_features(masks, masks, edges, queries=torch.empty(1, 0, 4),
                                             boxes=torch.empty(1, 0, 4), scores=torch.empty(1, 0))
    assert result["features"].shape == (1, 0, 20)
    output = ObservableRelation(appearance_channels=4, include_geometry=True, include_scores=True)(result)
    assert output["logits"].shape == (1, 0)
    with pytest.raises(ValueError):
        sparse_observable_pair_features(torch.ones(1, 2, 16, dtype=torch.bool),
                                         torch.ones(1, 2, 16, dtype=torch.bool), torch.tensor([[[0, 1]]]),
                                         queries=torch.full((1, 2, 4), float("inf")))
    with pytest.raises(ValueError):
        bounded_pair_edges(torch.empty(1, 129, 4))
