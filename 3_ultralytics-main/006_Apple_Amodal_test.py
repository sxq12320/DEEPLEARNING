from ultralytics import YOLO
from ultralytics.models.yolo.segment import SegmentationTrainer  # 你用的是seg
from rgb_d_dataset import RGBDDataset

# 正确注入点：覆盖 Trainer 里的 build_dataset 方法
class RGBDSegTrainer(SegmentationTrainer):
    def build_dataset(self, img_path, mode='train', batch=None):
        return RGBDDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=(mode == 'train'),
            hyp=self.args,
            rect=self.args.rect,
            cache=self.args.cache,
            single_cls=self.args.single_cls,
            stride=32,
            pad=0.5,
            prefix=mode,
            task='segment',  # ← 修改点：直接硬编码为 'segment' 或者使用 self.args.task
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == 'train' else 1.0,
            depth_dir='E:/mastercode/data/Apple_RGB_D_Amoal/depth_maps',  # ← 你的npy目录
        )

if __name__ == '__main__':
    args = dict(
        model=r'E:\mastercode\3_ultralytics-main\results\Amodal_Segment\Apple\3_yolo11n-seg-DWCONV_Any\weights\best.pt',
        data='E:/mastercode/3_ultralytics-main/206_Apple_Amodal.yaml',
        project=r'E:\mastercode\3_ultralytics-main\results\Amodal_Segment\Apple',
        name='3_yolo11n-seg-DWCONV_Any',
        epochs=50,
        imgsz=640,
        batch=8,
        max_det=100,
        # 严防死守色彩增强
        # --- 新增修改点：关闭多图增强操作 ---
        mosaic=0.0,  # 关闭马赛克增强（避免向空 buffer 索要数据，同时保护深度图几何特征）
        mixup=0.0,   # 关闭 MixUp 增强
        copy_paste=0.0, # 关闭复制粘贴增强（同理）
        hsv_h=0.0, 
        hsv_s=0.0, 
        hsv_v=0.0,
        
        workers=2,
    )
    trainer = RGBDSegTrainer(overrides=args)
    trainer.train()