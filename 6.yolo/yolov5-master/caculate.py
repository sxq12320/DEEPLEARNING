from utils.general import kmean_anchors
anchors = kmean_anchors('kouzhao.yaml', n=9, img_size=640)
print(anchors)  # 输出9个锚框的宽高