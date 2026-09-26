"""Synthetic mixed-aspect-ratio checks; no dataset writes or checkpoint downloads."""


def mixed_shape_batch(shapes=((640, 480), (480, 640)), empty_second=False):
    import numpy as np
    import torch
    from mmengine.structures import InstanceData
    from mmdet.structures import DetDataSample
    from mmdet.structures.mask import BitmapMasks

    inputs, samples = [], []
    for index, (height, width) in enumerate(shapes):
        count = 0 if empty_second and index == 1 else 2
        masks = np.zeros((count, height, width), dtype=np.uint8)
        boxes = []
        if count:
            # Include a one-pixel instance so padding cannot discard tiny masks.
            masks[0, 2, 3] = 1
            masks[1, 12:40, 15:45] = 1
            boxes = [[3, 2, 4, 3], [15, 12, 45, 40]]
        sample = DetDataSample(metainfo=dict(
            img_id=index, img_shape=(height, width), ori_shape=(height, width), scale_factor=(1., 1.)
        ))
        sample.gt_instances = InstanceData(
            bboxes=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            labels=torch.zeros(count, dtype=torch.long), masks=BitmapMasks(masks, height, width),
        )
        inputs.append(torch.zeros(3, height, width, dtype=torch.uint8))
        samples.append(sample)
    return dict(inputs=inputs, data_samples=samples)


def check_mask_padding(preprocessor):
    import numpy as np
    import torch

    report = []
    for shapes, empty in [(((640, 480), (480, 640)), False),
                          (((127, 95), (95, 127)), True)]:
        batch = mixed_shape_batch(shapes, empty_second=empty)
        originals = [sample.gt_instances.masks.masks.copy() for sample in batch["data_samples"]]
        boxes = [sample.gt_instances.bboxes.clone() for sample in batch["data_samples"]]
        output = preprocessor(batch, training=True)
        canvas = tuple(output["inputs"].shape[-2:])
        assert all(size % 32 == 0 for size in canvas), canvas
        tensors = []
        for sample, original, original_boxes in zip(output["data_samples"], originals, boxes):
            actual = sample.gt_instances.masks.masks
            height, width = original.shape[-2:]
            assert tuple(actual.shape[-2:]) == canvas, (actual.shape, canvas)
            assert np.array_equal(actual[:, :height, :width], original)
            assert actual.sum() == original.sum(), "Padding changed foreground pixels"
            assert torch.equal(sample.gt_instances.bboxes.cpu(), original_boxes)
            tensors.append(torch.as_tensor(actual))
        torch.cat(tensors, dim=0)  # Same size contract as RTMDetInsHead.loss_mask_by_feat.
        report.append(dict(input_shapes=shapes, canvas=canvas, empty_second=empty))
    # Validation masks stay at original resolution for evaluator coordinates.
    batch = mixed_shape_batch(((127, 95), (95, 127)))
    output = preprocessor(batch, training=False)
    assert [sample.gt_instances.masks.masks.shape[-2:] for sample in output["data_samples"]] == [(127, 95), (95, 127)]
    return report


def check_training_step(model, amp):
    import torch
    from mmengine.optim import build_optim_wrapper
    from mmdet_common import make_optim_wrapper_config

    model.train()
    model.zero_grad(set_to_none=True)
    wrapper_cfg = make_optim_wrapper_config(amp)
    if amp:
        # This disposable numerical check starts at scale 1 to avoid mistaking
        # GradScaler's normal initial-scale backoff for a broken model. Formal
        # training retains its original dynamic loss-scale configuration.
        wrapper_cfg["loss_scale"] = dict(init_scale=1.0)
    wrapper = build_optim_wrapper(model, wrapper_cfg)
    # A throwaway model created by preflight: never alters training initialization.
    batch = mixed_shape_batch(((128, 96), (96, 128)))
    with wrapper.optim_context(model):
        data = model.data_preprocessor(batch, training=True)
        losses = model(**data, mode="loss")
    total, logs = model.parse_losses(losses)
    if not bool(torch.isfinite(total)):
        raise RuntimeError(f"Non-finite synthetic MMDetection loss with AMP={amp}")
    wrapper.backward(total)
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(bool(torch.isfinite(grad).all()) for grad in gradients):
        raise RuntimeError(f"Non-finite/missing synthetic gradients with AMP={amp}")
    wrapper.step()
    wrapper.zero_grad()
    return {name: float(value.detach().cpu()) for name, value in logs.items()}


def check_prediction_step(model):
    import torch

    model.eval()
    batch = mixed_shape_batch(((127, 95), (95, 127)))
    with torch.no_grad():
        predictions = model.test_step(batch)
    report = []
    for sample, shape in zip(predictions, ((127, 95), (95, 127))):
        instances = sample.pred_instances
        if tuple(instances.masks.shape[-2:]) != shape:
            raise RuntimeError(f"Prediction masks not in original coordinates: {instances.masks.shape}, {shape}")
        if not bool(torch.isfinite(instances.scores).all()):
            raise RuntimeError("Non-finite synthetic prediction scores")
        report.append(dict(shape=shape, instances=len(instances)))
    return report


def check_large_mask_decode():
    """Exercise original-photo-sized masks, which tiny forward checks cannot cover."""
    import torch
    from mmdet_memory import decode_masks_bounded

    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    logits = torch.zeros(16, 80, 80, device="cuda")
    masks = decode_masks_bounded(logits, 8, True,
                                dict(ori_shape=(3072, 4096), scale_factor=(640 / 4096, 640 / 4096)), 0.5)
    assert tuple(masks.shape) == (16, 3072, 4096) and masks.device.type == "cpu"
    assert not masks.any()
    torch.cuda.synchronize()
    peak_mb = (torch.cuda.max_memory_allocated() - before) / 1024 ** 2
    return dict(mask_count=16, original_shape=(3072, 4096), extra_cuda_peak_MiB=round(peak_mb, 2))
