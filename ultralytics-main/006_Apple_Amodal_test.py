from ultralytics import YOLO
from ultralytics.nn.modules import C3k2_LS

if __name__ == '__main__':
     
    '''
    yolo = YOLO(r"ultralytics/cfg/models/11/yolo11n-seg.yaml") 
        1.使用最基本的yolo11nano架构,仅仅在网络输入部分增加一个深度信息,深度信息的样式为PNG格式

    yolo = YOLO(r"ultralytics-main/ultralytics/cfg/models/11/1_yolo11-seg-DWconv.yaml")
        2.在上面的基础之上,修改yolo11nano卷积效果,将原本的普通卷积变为现在的深度卷积+逐点卷积并进行运算
    
    yolo = YOLO(r"ultralytics-main/ultralytics/cfg/models/11/2_yolo11-seg-DWAny.yaml")
        3.在上面的基础之上,不仅仅修改普通的卷积,而且将C3K2模块内部的卷积全部替换成深度卷积+逐点卷积进行运算
    
    yolo = YOLO(r"code/ultralytics-main/ultralytics/cfg/models/11/3_yolo11-seg-RGBD_C3K2LS.yaml")
        4.在论文看大注意小的基础之上,我们引入了LSNET的架构,同时将LSNET和C3K2架构融合在一起并将其命名为C3K2_LS,在其他不发生修改的情况之下仅仅对模块进行替换
    
    yolo = YOLO(r"code/ultralytics-main/ultralytics/cfg/models/11/4_yolo11-seg-Dense-skiplink.yaml")
        5.直接就是修改yolo11nano的主干网络架构模型,引入自己创造的DenseSkip连接思路,并且将这个思路运用在主干网络之中实现带有层注意力的跳跃连接

    yolo = YOLO(r"code/ultralytics-main/ultralytics/cfg/models/11/5_yolo11_seg_Dense_and_C3K2LS.yaml")
        6.在上面的基础之上,也就是已经做过主干修改DenseSkip后东西,将其中的C3K2模块替换成C3K2_LS模块并进行运算
    
    yolo = YOLO(r"ultralytics/cfg/models/11/yolo11n-seg.yaml") 
        7.在原始的yolo11n以及深度信息進入的基礎之上將原本的AdamW優化器變爲現在的PIDAO優化器

    yolo = YOLO(r"/data/sxq/code/ultralytics-main/ultralytics/cfg/models/11/1_yolo11-seg-DWconv.yaml")
        8.在後面也就是2的基礎上將原本的AdamW優化器變爲現在的PIDAO優化器

    yolo = YOLO(r"code/ultralytics-main/ultralytics/cfg/models/11/2_yolo11-seg-DWAny.yaml")
        9.同樣也是將之前的全部換成深度可分裏卷積的哪一個版本的優化器變成PIDAO優化器

    yolo = YOLO(r"code/ultralytics-main/ultralytics/cfg/models/11/3_yolo11-seg-RGBD_C3K2LS.yaml")
        10.同樣在之前的基礎之上將C3K2_LS的網絡架構的優化器變成後面的PIDAO的優化器

    yolo = YOLO(r"code/ultralytics-main/ultralytics/cfg/models/11/4_yolo11-seg-Dense-skiplink.yaml")
        11.同樣在之前的基礎之上將DesneSkip爲主幹的網絡架構的優化器變成後面的PIDAO的優化器

    yolo = YOLO(r"code/ultralytics-main/ultralytics/cfg/models/11/5_yolo11_seg_Dense_and_C3K2LS.yaml")
        12.同樣在之前的基礎之上將C3K2_LS配合DesneSkip架構的網絡架構的優化器變成後面的PIDAO的優化器

    yolo = YOLO(r"code/ultralytics-main/ultralytics/cfg/models/11/0_yolo11-seg-RGBandD.yaml")
        13.在原本的基础之上使用AdamW优化器同时在不改变yolo11n的架构的条件之下增加一个圆形的形状先验
    
    '''
    # yolo = YOLO(r"code/ultralytics-main/ultralytics/cfg/models/11/0_yolo11-seg-RGBandD.yaml")
    yolo = YOLO(r"ultralytics-main/ultralytics/cfg/models/11/0_yolo11-seg-RGBandD.yaml")
    yolo.train(
        data=r'ultralytics-main/206_Apple_Amodal.yaml',
        project=r'ultralytics-main/results',
        name='12_yolo11n-seg-origin-circle-predicted',
        epochs=400,
        imgsz=640,
        batch=4,
        lr0=0.0001,
        momentum=0.9,
        weight_decay=0.0005,
        optimizer='AdamW',
        amp = False,
        cache=False,
        device = 0,
        workers = 4,

        # ellipse_param_weight = 0.01, # (float) ellipse parameter consistency loss gain (segment)
        # ellipse_dice_weight = 0.1, # (float) soft ellipse Dice prior loss gain (segment)
        # ellipse_softness = 8.0, # (float) boundary sharpness for soft ellipse rasterization (segment)
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
#         mosaic=0.0,  # 关闭马赛克增强（避免向空 buffer 索要数据,同时保护深度图几何特征）
#         mixup=0.0,   # 关闭 MixUp 增强
#         copy_paste=0.0, # 关闭复制粘贴增强（同理）
#         hsv_h=0.0, 
#         hsv_s=0.0, 
#         hsv_v=0.0,
        
#         workers=2,
#     )
#     trainer = RGBDSegTrainer(overrides=args)
#     trainer.train()

