from ultralytics import YOLO
from ultralytics.nn.modules import C3k2_LS

if __name__ == '__main__':
     
    
    '''
    yolo = YOLO(r"ultralytics/cfg/models/11/yolo11n-seg.yaml")
        1. 使用最基础的 yolo11nano 架构，仅在网络输入部分增加深度信息（PNG 格式）。

    yolo = YOLO(r"ultralytics-main/ultralytics/cfg/models/11/1_yolo11-seg-DWconv.yaml")
        2. 在上述基础上，将普通卷积替换为深度可分离卷积（Depthwise + Pointwise）。

    yolo = YOLO(r"ultralytics-main/ultralytics/cfg/models/11/2_yolo11-seg-DWAny.yaml")
        3. 进一步将 C3K2 模块内的所有卷积替换为深度可分离卷积。

    yolo = YOLO(r"code/ultralytics-main/ultralytics/cfg/models/11/3_yolo11-seg-RGBD_C3K2LS.yaml")
        4. 基于论文“看大注意小”的思想，引入 LSNET 架构，并将 LSNET 与 C3K2 融合，命名为 C3K2_LS；在不改变其余结构的情况下仅替换该模块。

    yolo = YOLO(r"code/ultralytics-main/ultralytics/cfg/models/11/4_yolo11-seg-Dense-skiplink.yaml")
        5. 修改 yolo11nano 的主干网络，提出 DenseSkip 跳接思想，在主干中实现带有层注意力的跳跃连接。

    yolo = YOLO(r"code/ultralytics-main/ultralytics/cfg/models/11/5_yolo11_seg_Dense_and_C3K2LS.yaml")
        6. 在完成主干 DenseSkip 修改后，将其中的 C3K2 模块替换为 C3K2_LS 并运行。

    yolo = YOLO(r"ultralytics/cfg/models/11/yolo11n-seg.yaml")
        7. 在原始 yolo11n（含深度信息输入）基础上，将优化器由 AdamW 替换为 PIDAO。

    yolo = YOLO(r"/data/sxq/code/ultralytics-main/ultralytics/cfg/models/11/1_yolo11-seg-DWconv.yaml")
        8. 在配置 2 的基础上，将优化器由 AdamW 替换为 PIDAO。

    yolo = YOLO(r"code/ultralytics-main/ultralytics/cfg/models/11/2_yolo11-seg-DWAny.yaml")
        9. 同样将对应配置中的优化器替换为 PIDAO（适用于深度可分离卷积版本）。

    yolo = YOLO(r"code/ultralytics-main/ultralytics/cfg/models/11/3_yolo11-seg-RGBD_C3K2LS.yaml")
        10. 在上述基础上，将 C3K2_LS 网络的优化器更换为 PIDAO。

    yolo = YOLO(r"code/ultralytics-main/ultralytics/cfg/models/11/4_yolo11-seg-Dense-skiplink.yaml")
        11. 在上述基础上，将以 DenseSkip 为主干的网络的优化器更换为 PIDAO。

    yolo = YOLO(r"code/ultralytics-main/ultralytics/cfg/models/11/5_yolo11_seg_Dense_and_C3K2LS.yaml")
        12. 在上述基础上，将 C3K2_LS 与 DenseSkip 结合的网络的优化器更换为 PIDAO。

    yolo = YOLO(r"code/ultralytics-main/ultralytics/cfg/models/11/0_yolo11-seg-RGBandD.yaml")
        13. 在保持 yolo11n 架构不变的前提下，使用 AdamW 优化器，并增加一个圆形形状先验。
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
    )