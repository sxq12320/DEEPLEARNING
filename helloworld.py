from typing import List, Optional, Union

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

matplotlib.rc("font", family="Microsoft YaHei")


def img_show(
    *imgs: Union[np.ndarray, Image.Image],
    titles: Optional[List[str]] = None,
    cmap: Optional[str] = None,
    figsize: Optional[tuple] = None,
    axis: str = "off",
    max_cols: int = 4,
    save_path: Optional[str] = None,
) -> None:
    """
    Using : 使用plt对多张照片进行展示全自动模型

    ---
    Args:
        *imgs : np.ndarray 或 PIL.Image
            任意数量的图片，支持灰度、RGB、RGBA。
        titles : list of str, optional
            每张图片的标题，长度需与图片数量一致。
        cmap : str, optional
            颜色映射，默认灰度图自动使用 'gray'，彩色图忽略。
        figsize : tuple, optional
            图像整体大小 (宽, 高)，默认自动计算。
        axis : str, {'on', 'off'}
            是否显示坐标轴。
        max_cols : int
            每行最多显示的子图列数。
        save_path : str, optional
            若提供，将图片保存到该路径，不显示 GUI。

    ---
    Returns : None

    """
    # 全部转换为 numpy 数组
    arrays = []
    for img in imgs:
        if isinstance(img, Image.Image):
            img = np.array(img)
        elif isinstance(img, np.ndarray):
            pass
        else:
            raise TypeError(
                f"不支持的类型：{type(img)}，仅仅支持 np.ndarray 或者是 PIL.Image"
            )
        arrays.append(img)

    # 判定图片是否存在问题
    n = len(arrays)
    if n == 0:
        print("there is no picture")
        return

    # 判定每一个图片的数量大小长宽分布
    ncols = min(n, max_cols)
    nrows = (n + ncols - 1) // ncols

    # 自动计算 figsize 如果没有指定的话
    if figsize is None:
        scale = 3
        figsize = (ncols * scale, nrows * scale)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)
    for idx, arr in enumerate(arrays):
        ax = axes[idx]
        # 归一化处理：如果数据是 0-255 整数且超过1，自动转为 0-1 浮点
        if arr.dtype == np.uint8:
            display_arr = arr
        elif arr.max() > 1.0 and arr.dtype.kind == "f":
            display_arr = (arr * 255).clip(0, 255).astype(np.uint8)
        else:
            display_arr = arr
        # 判断是否为单通道（灰度）
        is_grayscale = False
        if display_arr.ndim == 2:
            is_grayscale = True
        elif display_arr.ndim == 3 and display_arr.shape[2] == 1:
            display_arr = display_arr.squeeze(axis=2)
            is_grayscale = True
        elif display_arr.ndim == 3 and display_arr.shape[2] == 2:
            # 特殊处理：两通道视为灰度加掩码？这里简单当灰度显示第一通道
            display_arr = display_arr[:, :, 0]
            is_grayscale = True

        # 确定颜色映射
        if cmap is not None:
            use_cmap = cmap
        elif is_grayscale:
            use_cmap = "gray"
        else:
            use_cmap = None

        # 显示
        if is_grayscale or use_cmap is not None:
            ax.imshow(display_arr, cmap=use_cmap)
        else:
            ax.imshow(display_arr)

        # 设置标题
        if titles and idx < len(titles):
            ax.set_title(titles[idx])

        ax.axis(axis)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"图片已保存至 {save_path}")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    img_path = r"C:\Users\33836\Desktop\test.png"
    # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR_BGR)
    img = cv2.resize(img, (512, 512))

    f = np.float32(img)
    F = np.fft.fft2(f)  # 二维傅里叶变换
    F_shiffted = np.fft.fftshift(F)  # 移动到中心位置

    magnitude = np.abs(F_shiffted)
    magnitude_log = 20 * np.log1p(magnitude)  # 取对数
    magnitude_display = magnitude_log / np.max(magnitude_log)  # 归一化
    phase = np.angle(F_shiffted)  # 相频谱
    phase_display = (phase + np.pi) / (2 * np.pi)  # 归一化

    img_show(
        img,
        magnitude_display,
        phase_display,
        titles=["原始图像", "傅里叶幅频谱", "傅里叶相频谱"],
    )
