from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 加载你刚刚训练好的“学霸脑子”
    # 【注意】：请把下面这个路径替换成你实际 best.pt 的路径！
    model = YOLO(r'E:\mastercode\runs\detect\train5\weights\best.pt') 

    # 2. 开启摄像头实时检测
    print("正在启动摄像头，请准备好你的口罩...")
    
    # source='0' 代表使用电脑自带的默认摄像头 (如果是外接USB摄像头，可以试试 '1')
    # show=True 代表直接弹出窗口实时显示检测画面
    # conf=0.5 代表置信度阈值（只有模型觉得大于50%概率的框才会显示出来，过滤掉瞎猜的框）
    results = model.predict(source='0', show=True, conf=0.5)