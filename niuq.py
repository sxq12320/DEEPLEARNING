import numpy as np
from PIL import Image
import math

def generate_escher_image(input_path, output_path, scale_factor=2.0, output_size=800, arms=1):
    """
    使用复变函数原理复现 M.C. Escher 的 Print Gallery 扭曲效果
    
    参数:
    - input_path: 输入原图的路径 (建议使用正方形图片)
    - output_path: 生成图片的保存路径
    - scale_factor: 缩放因子 s (原图相框相对于整体图片的缩小倍数)
    - output_size: 输出图片的分辨率大小
    - arms: 螺旋的臂数 (控制旋转的剧烈程度，默认为 1)
    """
    
    # 1. 加载图片并转换为 Numpy 数组
    img = Image.open(input_path).convert("RGB")
    img_data = np.array(img)
    height, width, channels = img_data.shape

    # 2. 创建输出图片的空白数组
    out_data = np.zeros((output_size, output_size, channels), dtype=np.uint8)

    # 3. 定义数学变换的核心系数 β (Beta)
    # 修正：根据数学家 Hendrik Lenstra 解析埃舍尔画作的公式
    # 要想让图像在目标平面旋转一圈 (2πi) 时无缝衔接，同时完成 s 倍的缩放 (ln s)
    # 且形成指定数量的螺旋臂 (arms)，必须满足变换： beta * 2πi = arms * 2πi + ln(s) 
    # 即： beta = arms - i * (ln(s) / 2π)
    s = scale_factor
    beta = arms - 1j * (math.log(s) / (2 * np.pi))
    # 注：如果想反方向旋转，可以将减号改为加号

    # 4. 在复平面上为目标图片构建坐标网格
    # 将输出像素的坐标归一化到 [-1, 1] 范围
    x = np.linspace(-1, 1, output_size)
    y = np.linspace(-1, 1, output_size)
    X, Y = np.meshgrid(x, y)
    
    # 计算机图像坐标的 Y 轴是向下的，所以在复数中表现为减去 1j * Y
    Z_target = X - 1j * Y

    # 防止中心点出现 log(0) 的数学错误
    Z_target[Z_target == 0] = 1e-10 + 1e-10j

    # 5. 核心数学映射： Z_src = Z_target ^ β
    # 我们利用指数和对数形式来计算： exp(beta * log(Z_target))
    Z_src = np.exp(beta * np.log(Z_target))

    # 6. 对生成的 Z_src 进行对数尺度的周期包裹
    r = np.abs(Z_src)
    theta = np.angle(Z_src)

    # 获取对数尺度的周期小数部分 t (范围 [0, 1))
    t = (np.log(r) / np.log(s)) % 1
    
    # 计算平滑的混合权重 (使用余弦平滑，让内外边缘无缝融合)
    # 当 t 趋向 0 时，w 趋向 0；当 t 趋向 1 时，w 趋向 1
    w = (1 - np.cos(np.pi * t)) / 2
    w = w[..., np.newaxis] # 扩展一个维度以匹配 RGB 的 3 通道

    # 分别计算两个相邻缩放周期的极径
    r1 = s ** t         # 当前周期，范围 [1, s)
    r2 = s ** (t - 1)   # 内部相邻周期，范围 [1/s, 1)

    # 7. 将两个周期的极坐标转换回直角坐标系
    X1, Y1 = r1 * np.cos(theta), r1 * np.sin(theta)
    X2, Y2 = r2 * np.cos(theta), r2 * np.sin(theta)

    # 8. 映射回原始图片的像素坐标 
    # 为了让任意形状的图片都能用，我们将采样的最大半径设为图片最短边的一半
    R_max = min(width, height) / 2
    scale_img = R_max / s  # 归一化系数

    def get_pixels(X_src, Y_src):
        px = X_src * scale_img + (width / 2)
        py = -Y_src * scale_img + (height / 2)
        px = np.clip(px.astype(int), 0, width - 1)
        py = np.clip(py.astype(int), 0, height - 1)
        return img_data[py, px]

    # 分别对两个周期进行采样
    color1 = get_pixels(X1, Y1)
    color2 = get_pixels(X2, Y2)

    # 9. 采样并赋值：按权重将两个周期渐变融合，消除任何突兀的接缝
    out_data = ((1 - w) * color1 + w * color2).astype(np.uint8)

    # 10. 保存生成的图像
    out_img = Image.fromarray(out_data)
    out_img.save(output_path)
    print(f"成功生成埃舍尔扭曲效果图片，已保存至: {output_path}")

if __name__ == "__main__":
    # 使用示例 (请确保同目录下有一张名为 test.jpg 的图片进行测试)
    # scale_factor 越大，或者 arms 越大，螺旋的扭曲旋转感越强！
    try:
        generate_escher_image(
            input_path="test.jpg", 
            output_path="escher_output.jpg", 
            scale_factor=10.0,  # 增大缩放倍数，让旋转效果更明显
            output_size=1000,
            arms=1              # 螺旋臂的数量，可以尝试改为 2 或 3
        )
    except FileNotFoundError:
        print("请提供一张名为 test.jpg 的图片放在当前目录中运行。")