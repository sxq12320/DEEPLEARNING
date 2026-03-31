from PIL import Image
import numpy as np
import cv2
import torch

def keep_image_size_open(path , size = (256,256)):
    '''
    using:
        将图片大小进行统一
        1.获取图片中的最大变成
        2.生成最大边×最大边的掩码mask
        3.将原图粘贴到掩码的左上角
        4.将图像进行缩放
    Args:
        path (str) : 图片的地址
        size (list) : 调整后的图片大小
    Returns:
        mask(Lise) : 完成后的一张图片
    '''
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('RGB' , (temp , temp) , (0 , 0 , 0))
    mask.paste(img , (0 , 0))
    mask = mask.resize(size)
    return mask

def enhance_image (img:Image.Image) ->Image.Image:
    '''
    using:
        对原始的图像进行增强处理
        1.滤波去噪操作
        2.局部对比度增强
        3.使用边缘增强操作
    Args:
        img ： 普通的三原色图像
    Returns：
        result ： 增强后的三原色图像
    '''

    # 双边滤波去噪
    img_np = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    denoised = cv2.bilateralFilter(
        img_bgr,
        d=7,  # 滤波邻域直径，越大越慢，7~9 是常用值
        sigmaColor=50,  # 颜色空间标准差：越大，越远的颜色也会被混合
        sigmaSpace=50  # 坐标空间标准差：越大，越远的像素也参与
    )

    #局部对比度增强处理
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=2.0,  # 对比度限制，防止过度放大噪声；1.5~3.0 之间调
        tileGridSize=(8, 8)  # 局部块大小；图像越小可以适当改成 (4,4)
    )
    l_enhanced = clahe.apply(l)

    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # 进行边缘锐化操作
    blurred = cv2.GaussianBlur(enhanced_bgr, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(
        enhanced_bgr, 1.5,  # 原图权重
        blurred, -0.5,  # 减去模糊图权重（strength = 0.5，可以调大到 1.0）
        0, 0
    )
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    # 修改回原本的三原色图片返回
    result_rgb = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_rgb)

# utils.py 新增
def keep_seg_size_open(path, size=(256, 256)):
    img = Image.open(path)  # 保持原始 P 模式，不转 RGB
    temp = max(img.size)
    mask = Image.new('P', (temp, temp), 0)  # 背景填 0
    mask.paste(img, (0, 0))
    mask = mask.resize(size, Image.NEAREST)  # ⚠️ 必须用 NEAREST，不能插值类别ID
    return mask