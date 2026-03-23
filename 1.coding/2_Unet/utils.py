from PIL import Image

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

