"""TS-Dual 架构默认配置。"""

TS_DUAL_MODEL_CFG = {
    "backbone": {
        "name": "ts_dual_backbone",
        "args": {
            "in_ch_rgb": 4,
            "in_ch_depth": 1,
            "channels": [32, 64, 128],
            "activation": "silu",
            "exchange_reduction": 4,
        },
    },
    "neck": [
        {
            "name": "afpn_neck",
            "args": {
                "in_channels": [32, 64, 128],
                "out_channels": 128,
                "activation": "silu",
            },
        },
        {
            "name": "dyhead_neck",
            "args": {
                "channels": 128,
                "reduction": 4,
                "activation": "silu",
            },
        },
    ],
    "head": {
        "name": "decoupled_segdet_head",
        "args": {
            "in_channels": 128,
            "mask_out_ch": 1,
            "activation": "silu",
            "bbox_hidden": 128,
        },
    },
}

TS_DUAL_LOSS_CFG = {
    "mask_loss": "bce",
    "mask_weight": 1.0,
    "fourier_weight": 0.2,
    "bbox_weight": 1.0,
    "nwd_constant": 20.0,
}
