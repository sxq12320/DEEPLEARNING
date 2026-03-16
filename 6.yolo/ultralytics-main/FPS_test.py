import time
import torch
import numpy as np
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'E:\mastercode\6.yolo\runs\segment\train5\weights\best.pt')  # 换成你的实际路径

    # 预热
    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    for _ in range(10):
        model(dummy, verbose=False)

    # 测速
    runs = 100
    start = time.perf_counter()
    for _ in range(runs):
        model(dummy, verbose=False)
    end = time.perf_counter()

    avg_ms = (end - start) / runs * 1000
    fps = 1000 / avg_ms

    print(f"平均推理时间: {avg_ms:.2f} ms")
    print(f"FPS: {fps:.1f}")
    print(f"设备: {'GPU - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")