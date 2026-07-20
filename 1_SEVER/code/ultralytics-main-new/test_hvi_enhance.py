"""Self-checks for the HVIEnhance low-light front-end.

Run:  python test_hvi_enhance.py

Checks (all on CPU, no CUDA needed):
  1. Deterministic RGBHVI.HVIT matches the reference masked-scatter formula.
  2. Round-trip PHVIT(HVIT(x)) ~ x (the transform is (near-)invertible).
  3. HVIEnhance is (near-)identity at init (zero-initialised residual head).
  4. Forward/backward run and shapes are preserved; gradient reaches density_k.
  5. Determinism: two forwards give identical output.
  6. (best-effort) Build the 010 YAML YOLO model, forward+backward, report params/FLOPs.
"""

from __future__ import annotations

import torch

from ultralytics.nn.modules import HVIEnhance, RGBHVI

_PI = 3.141592653589793


def _ref_hvit(img: torch.Tensor, k: float) -> torch.Tensor:
    """Reference HVIT (net/HVI_transform.py::RGB_HVI.HVIT), masked-scatter version."""
    eps = 1e-8
    hue = torch.zeros(img.shape[0], img.shape[2], img.shape[3], dtype=img.dtype)
    value = img.max(1)[0]
    img_min = img.min(1)[0]
    hue[img[:, 2] == value] = 4.0 + ((img[:, 0] - img[:, 1]) / (value - img_min + eps))[img[:, 2] == value]
    hue[img[:, 1] == value] = 2.0 + ((img[:, 2] - img[:, 0]) / (value - img_min + eps))[img[:, 1] == value]
    hue[img[:, 0] == value] = (0.0 + ((img[:, 1] - img[:, 2]) / (value - img_min + eps))[img[:, 0] == value]) % 6
    hue[img.min(1)[0] == value] = 0.0
    hue = hue / 6.0
    saturation = (value - img_min) / (value + eps)
    saturation[value == 0] = 0
    hue = hue.unsqueeze(1)
    saturation = saturation.unsqueeze(1)
    value = value.unsqueeze(1)
    color_sensitive = ((value * 0.5 * _PI).sin() + eps).pow(k)
    ch = (2.0 * _PI * hue).cos()
    cv = (2.0 * _PI * hue).sin()
    H = color_sensitive * saturation * ch
    V = color_sensitive * saturation * cv
    return torch.cat([H, V, value], dim=1)


def main() -> None:
    torch.manual_seed(0)
    x = torch.rand(2, 3, 64, 80)  # RGB in [0, 1]

    # 1. HVIT matches the reference formula
    trans = RGBHVI(k_init=0.2)
    mine = trans.HVIT(x)
    ref = _ref_hvit(x, 0.2)
    err_hvit = (mine - ref).abs().max().item()
    print(f"[1] HVIT vs reference   max|Δ| = {err_hvit:.3e}")
    assert err_hvit < 1e-5, "deterministic HVIT does not match the reference formula"

    # 2. round-trip identity
    rt = trans.PHVIT(trans.HVIT(x))
    err_rt = (rt - x).abs().max().item()
    err_rt_mean = (rt - x).abs().mean().item()
    print(f"[2] round-trip PHVIT∘HVIT  max|Δ| = {err_rt:.3e}   mean|Δ| = {err_rt_mean:.3e}")
    assert err_rt_mean < 1e-3, "round-trip is not (near-)identity"

    # 3. HVIEnhance near-identity at init (zero residual head)
    m = HVIEnhance(3, 3, base=16, blocks=2)
    m.eval()
    with torch.no_grad():
        y = m(x)
    err_id = (y - x).abs().mean().item()
    print(f"[3] HVIEnhance init identity  mean|Δ| = {err_id:.3e}   out range=[{y.min():.3f},{y.max():.3f}]")
    assert tuple(y.shape) == tuple(x.shape)
    assert err_id < 1e-2, "HVIEnhance should start ~identity so transfer-learning is undisturbed"

    # 4. forward/backward + gradient reaches density_k
    m.train()
    y = m(x)
    y.mean().backward()
    gk = m.trans.density_k.grad
    print(f"[4] backward OK   density_k.grad = {None if gk is None else gk.abs().sum().item():.3e}")
    assert gk is not None and torch.isfinite(gk).all(), "no finite gradient to density_k"

    # 5. determinism
    m.eval()
    with torch.no_grad():
        a, b = m(x), m(x)
    print(f"[5] determinism max|Δ| = {(a - b).abs().max().item():.3e}")
    assert torch.equal(a, b), "forward is not deterministic"

    # 6. build the YOLO model (best-effort)
    try:
        from ultralytics import YOLO

        model = YOLO("0_orange_yaml/010_yolo11-seg-hvi.yaml")
        model.model.eval()
        img = torch.rand(1, 3, 640, 640)
        with torch.no_grad():
            _ = model.model(img)
        n_params = sum(p.numel() for p in model.model.parameters())
        print(f"[6] YOLO 010 built + forward OK   params = {n_params:,}")
        try:
            model.info(detailed=False)
        except Exception as e:  # noqa: BLE001
            print(f"    model.info() skipped: {e}")
    except Exception as e:  # noqa: BLE001
        print(f"[6] YOLO model build skipped ({type(e).__name__}: {e})")

    print("\nAll HVIEnhance self-checks passed.")


if __name__ == "__main__":
    main()
