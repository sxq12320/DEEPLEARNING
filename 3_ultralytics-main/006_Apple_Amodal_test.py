from ultralytics import YOLO
from ultralytics.nn.modules import C3k2_LS

if __name__ == '__main__':
     
    '''
    yolo = YOLO(r"ultralytics/cfg/models/11/yolo11n-seg.yaml") 
        RGB three channels and a Depth Channel with the basic yolo11n which is not changed

    yolo = YOLO(r"ultralytics-main/ultralytics/cfg/models/11/1_yolo11-seg-DWconv.yaml")
        RGB three channels and a depth channel with the basic yolo11n which is changed Conv modules in the backbone but not changed the C3K2 module
    
    yolo = YOLO(r"ultralytics-main/ultralytics/cfg/models/11/2_yolo11-seg-DWAny.yaml")
        RGB and Depth four channels with the basic architechure yolo11n  which is changed C3K2 module with the C3K2_DW.
        In the C3K2_DW, all Conv module were replaced with depth wide Conv and pointed wide Conv
    
    yolo = YOLO(r"code/ultralytics-main/ultralytics/cfg/models/11/3_yolo11-seg-RGBD_C3K2LS.yaml")
        using the module from some article named "See large attention small".then upated the C3K2 module from the basic to C3K2_LS.
        In mean time , we using the yolo 11 nano backbone to train this amodal segmentation tasks 
    
    yolo = YOLO(r"code/ultralytics-main/ultralytics/cfg/models/11/4_yolo11-seg-Dense-skiplink.yaml")
        using my famous Backbone which named Dense Skip .this backbone is added a C3K2 module in the basic yolo11 backbone.
        in the mean time, we using the such resnet to concat feature better

    
        
    '''
    yolo = YOLO(r"3_ultralytics-main/ultralytics/cfg/models/11/5_yolo11_seg_Dense_and_C3K2LS.yaml")
    yolo.train(
        data=r'3_ultralytics-main/206_Apple_Amodal.yaml',
        project=r'3_ultralytics-main/results/Amodal_Segment/Apple',
        name='5_yolo11n-seg-Dense-and-C3K2_ls',
        epochs=500,
        imgsz=640,
        batch=2,
        lr0=0.0001,
        momentum=0.9,
        weight_decay=0.0005,
        optimizer='AdamW',
        amp = False,
        cache=True,
        device = 0,
        workers=2,
    )






















# from ultralytics import YOLO
# from ultralytics.models.yolo.segment import SegmentationTrainer  # 你用的是seg
# from rgb_d_dataset import RGBDDataset,RGBDDataset_pic

# # # 正确注入点：覆盖 Trainer 里的 build_dataset 方法
# # class RGBDSegTrainer(SegmentationTrainer):
# #     def build_dataset(self, img_path, mode='train', batch=None):
# #         return RGBDDataset_pic(
# #             img_path=img_path,
# #             imgsz=self.args.imgsz,
# #             batch_size=batch,
# #             augment=(mode == 'train'),
# #             hyp=self.args,
# #             rect=self.args.rect,
# #             cache=self.args.cache,
# #             single_cls=self.args.single_cls,
# #             stride=32,
# #             pad=0.5,
# #             prefix=mode,
# #             task='segment',  # ← 修改点：直接硬编码为 'segment' 或者使用 self.args.task
# #             classes=self.args.classes,
# #             data=self.data,
# #             fraction=self.args.fraction if mode == 'train' else 1.0,
# #             depth_dir='E:/mastercode/data/Apple_RGB_D_Amoal/depth_pic',  # ← 你的npy目录
# #         )

# if __name__ == '__main__':
#     args = dict(
#         model=r'E:\mastercode\3_ultralytics-main\ultralytics\cfg\models\11\yolo11n-seg.yaml',
#         data='E:/mastercode/3_ultralytics-main/206_Apple_Amodal.yaml',
#         project=r'E:\mastercode\3_ultralytics-main\results\Amodal_Segment\Apple',
#         name='5_yolo11n-seg-DA-prediceted',
#         epochs=300,
#         imgsz=640,
#         batch=8,
#         max_det=100,
#         # 严防死守色彩增强
#         # --- 新增修改点：关闭多图增强操作 ---
#         mosaic=0.0,  # 关闭马赛克增强（避免向空 buffer 索要数据，同时保护深度图几何特征）
#         mixup=0.0,   # 关闭 MixUp 增强
#         copy_paste=0.0, # 关闭复制粘贴增强（同理）
#         hsv_h=0.0, 
#         hsv_s=0.0, 
#         hsv_v=0.0,
        
#         workers=2,
#     )
#     trainer = RGBDSegTrainer(overrides=args)
#     trainer.train()

