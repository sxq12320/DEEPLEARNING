from ultralytics import YOLO
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.cfg import get_cfg, DEFAULT_CFG
from FEM_DCN import replace_c2f_with_fem, collect_fem_loss


class FEMSegmentationTrainer(SegmentationTrainer):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        self.lambda_fem = overrides.pop('lambda_fem', 0.05)
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        print("\n[FEM] Replacing C2f modules...")
        replace_c2f_with_fem(model)
        return model

    def criterion(self, preds, batch):
        seg_loss, loss_items = super().criterion(preds, batch)
        fem_loss = collect_fem_loss(self.model)
        total_loss = seg_loss + self.lambda_fem * fem_loss
        return total_loss, loss_items


if __name__ == '__main__':
    yolo = YOLO('yolo11n-seg.pt')
    yolo.train(
        data=r'E:\\mastercode\\6.yolo\\ultralytics-main\\voc20007_seg_yolov8.yaml',
        project=r'E:/mastercode/6.yolo/runs/segment',
        epochs=300,
        imgsz=640,
        batch=8,
        # lr0 = 0.00001,
        # lrf = 0.01,
        # momentum = 0.73,
        # weight_decay = 0.0005,
        # warmup_epochs = 3,
        # optimizer='AdamW',
        device=0  # CPU 则改为 'cpu'
    )



# if __name__ == '__main__':
#     trainer = FEMSegmentationTrainer(
#         overrides={
#             'model':      'yolo11n-seg.pt',
#             'data':       r'E:/mastercode/6.yolo/ultralytics-main/voc20007_seg_yolov8.yaml',
#             'project':    r'E:/mastercode/6.yolo/runs/segment',
#             'epochs':     300,
#             'imgsz':      640,
#             'batch':      8,
#             'device':     0,
#             'lambda_fem': 0.05,
#         }
#     )
#     trainer.train()