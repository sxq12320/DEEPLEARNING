"""
SMC (Sliding Mode Control) Optimizer 测试脚本
基于 YOLO11-seg，使用 SMCScheduler 替代 AdamW

超参数说明：
  smc_plateau_patience  : loss 不改善的 epoch 数后触发 plateau
  smc_lr_boost          : plateau 时 LR 提升倍数
  smc_noise_scale       : 梯度噪声标准差
  smc_beta1_low         : plateau 时 β₁
"""

from ultralytics import YOLO

if __name__ == '__main__':
    yolo = YOLO(r'E:\mastercode\6.yolo\ultralytics-main\ultralytics\cfg\models\11\yolo11n-seg.yaml')
    yolo.train(
        data=r'E:\mastercode\6.yolo\ultralytics-main\201_caomei_data.yaml',
        project=r'E:/mastercode/6.yolo/runs/segment',
        name='smc_test',
        epochs=100,
        imgsz=512,
        batch=8,
        lr0=0.001,
        momentum=0.937,
        weight_decay=0.0005,
        optimizer='SMC',
        device=0,
        # ---- SMC 超参数 (可选，不传则用默认值) ----
        smc_surface_patience=10,
        smc_lr_boost=1.2,
        smc_noise_scale=0.003,
        smc_beta1_low=0.85,
        # ---- 数据增强 ----
        copy_paste=0.3,
        erasing=0.3,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        scale=0.5,
        degrees=15,
        flipud=0.3,
        translate=0.15,
        mosaic=0.8,
    )
