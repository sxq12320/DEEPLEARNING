"""
SMC (Sliding Mode Control) Optimizer 测试脚本
基于 YOLO11-seg，使用 SMCScheduler 替代 AdamW

超参数说明：
  smc_plateau_threshold  : 梯度停滞检测阈值 (grad_norm 均值/峰值 < 此值 → plateau)
  smc_plateau_patience   : plateau 持续多少步后触发逃离
  smc_escape_push        : 逃离时每步参数位移量
  smc_escape_push_steps  : 逃离推进步数 (总位移 = push × steps)
  smc_reconv_steps       : 逃离后快速重收敛步数
  smc_reconv_lr_mult     : 重收敛时 LR 放大倍数
  smc_beta1_low          : 逃离/重收敛时的 β₁
  smc_beta2_low          : 逃离/重收敛时的 β₂
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
        smc_plateau_threshold=0.15,   # 梯度停滞阈值
        smc_plateau_patience=5,       # plateau 耐心值
        smc_escape_push=0.10,         # 逃离位移量
        smc_escape_push_steps=20,     # 逃离步数
        smc_reconv_steps=40,          # 重收敛步数
        smc_reconv_lr_mult=3.0,       # 重收敛 LR 倍数
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
