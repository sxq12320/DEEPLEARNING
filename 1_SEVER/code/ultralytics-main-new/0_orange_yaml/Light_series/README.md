# Light series v3

Light v3 is the final G0830-informed lightweight experiment matrix. It is not a
collection of attention blocks. The structural study changes only two factors:

1. the deep P4/P5 backbone stage (`CitrusLightStage`, `CitrusRepMixerStage`, or
   official `C3k2`); and
2. the neck topology (official PAN or residual `CitrusLightAFPN`).

All structural YAMLs retain the official shallow P2/P3 stages, SPPF, C2PSA,
four-scale P2--P5 prediction, and the standard Segment head unless the filename
explicitly says `lite` or `quality`. This removes the hidden C2PSA/head/loss
confounds that affected earlier Light drafts.

## Structure screen

| Model | Deep P4/P5 | Neck | Head | Controlled question |
|---|---|---|---|---|
| Light00 | PConv residual stage | official PAN | Segment | PConv backbone effect only |
| Light01 | official C3k2 | residual AFPN | Segment | AFPN topology effect only |
| Light02 | PConv residual stage | residual AFPN | Segment | PConv x AFPN interaction |
| Light05 | RepMixer stage | official PAN | Segment | clean G04-supported backbone isolation |
| Light06 | RepMixer stage | residual AFPN | Segment | RepMixer x AFPN interaction |

## Pareto candidates

| Model | Structure | Purpose |
|---|---|---|
| Light03 | PConv + AFPN + lite head | aggressive speed/size candidate |
| Light04 | same Light03 graph + mask-quality branch | isolate score calibration |
| Light07 | RepMixer + official PAN + lite head | conservative accuracy-preserving candidate |

`Light04_quality_rank.yaml` requires `citrus_quality=0.25`; the batch runner
applies and records it. Light03 and Light04 differ only in their quality branch
and its loss setting.

## PR-specific controlled queue

The `--suite pr` queue keeps the exact Light03 graph for BCE, VFL, NWD, and
VFL+NWD. Only LightP04 changes to the Light04 quality head.

| Run | Only intended change |
|---|---|
| LightP00 | stock BCE control |
| LightP01 | `citrus_vfl=0.25` |
| LightP02 | `nwd_ratio=0.25` |
| LightP03 | NWD + VFL interaction |
| LightP04 | explicit mask-IoU quality branch |

The standard Ultralytics PR plot appends zero precision after the maximum
achieved recall. The terminal vertical cliff is therefore an AP integration
sentinel. Judge improvements by the recall ceiling, precision within the
achievable recall range, raw TP/FP/FN, and AP, not by whether the plotted line
touches zero at recall 1.

## Commands

```bash
python 20260830_citrus_light_batch.py --data /path/data.yaml --suite smoke --epochs 3 --device 0
python 20260830_citrus_light_batch.py --data /path/data.yaml --suite screen --epochs 50 --device 0
python 20260830_citrus_light_batch.py --data /path/data.yaml --suite pareto --epochs 50 --device 0
python 20260830_citrus_light_batch.py --data /path/data.yaml --suite pr --epochs 50 --device 0
```

Use `best.pt`. Completed G0830 models peaked around epochs 54--87 and generally
degraded by epoch 300, so run the 50-epoch screen first, promote only competitive
models to 100 epochs, and reserve 300 epochs/three seeds for the final method.
