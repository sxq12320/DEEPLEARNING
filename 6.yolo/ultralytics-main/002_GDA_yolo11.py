import torch
import torch.multiprocessing as mp
from ultralytics import YOLO

if __name__ == '__main__':
    mp.freeze_support()
    
    # ← 关键：关闭确定性模式，允许非确定性CUDA算子
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
    yolo = YOLO(r'E:\mastercode\6.yolo\ultralytics-main\ultralytics\cfg\models\11\yolo11-seg-tansconv.yaml')
    yolo.train(
        data=r'E:\mastercode\6.yolo\ultralytics-main\201_caomei_data.yaml',
        project=r'E:/mastercode/6.yolo/runs/segment',
        epochs=150,
        imgsz=512,
        batch=8,
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        optimizer='SGD',
        device=0,
        amp=False,
        workers=0,
        cache=False,
        deterministic=False,  # ← 同时在训练参数里也关掉
        seed=0,
    )

# from ultralytics import YOLO
# import torch.multiprocessing as mp
# import torch

# if __name__ == '__main__':
#     mp.freeze_support()
    
#     print(">>> 第1步：开始加载模型...")
#     yolo = YOLO(r'E:\mastercode\6.yolo\ultralytics-main\ultralytics\cfg\models\11\yolo11-seg-tansconv.yaml')
#     print(">>> 第2步：模型加载完成")
    
#     print(">>> 第3步：验证前向传播...")
#     dummy = torch.randn(1, 3, 512, 512).cuda()
#     model = yolo.model.cuda().eval()
#     with torch.no_grad():
#         out = model(dummy)
#     print(f">>> 第4步：前向传播成功，输出类型: {type(out)}")
#     del dummy, out
#     torch.cuda.empty_cache()
    
#     print(">>> 第5步：开始训练...")
#     yolo.train(
#         data=r'E:\mastercode\6.yolo\ultralytics-main\201_caomei_data.yaml',
#         project=r'E:/mastercode/6.yolo/runs/segment',
#         epochs=150,
#         imgsz=512,
#         batch=1,
#         lr0=0.01,
#         momentum=0.937,
#         weight_decay=0.0005,
#         optimizer='SGD',
#         device=0,
#         amp=False,
#         workers=0,
#         cache=False,
#     )
#     print(">>> 第6步：训练完成")