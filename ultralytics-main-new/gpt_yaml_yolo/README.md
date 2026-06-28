# GPT YOLO 改进配置说明

本目录保存针对西瓜花实例分割的一阶段 YOLO 改进 YAML。设计思路参考“论文创新指南”中的 A+B+C 组合策略，但只使用当前仓库已经注册并可直接解析的模块，避免训练时报模块缺失或参数不匹配。

## 配置文件

- `gpt_yolo11n_seg_p2_light_abc.yaml`：偏轻量。A=P2 小目标分割头，B=`C3Ghost` 轻量特征块，C=`SCDown` 轻量下采样，D=更小的 `Segment [nc, 24, 192]` 掩码头。适合先跑速度、参数量和二阶段联动。
- `gpt_yolo11n_seg_p2_psa_abcd.yaml`：偏精度。A=P2 小目标分割头，B=`SCDown` 降低下采样成本，C=`C2PSA` 低分辨率注意力增强，D=完整 `Segment [nc, 32, 256]` 掩码头。适合作为主实验精度版。

## 推荐训练顺序

先跑轻量版确认数据、训练流程和二阶段接口稳定：

```powershell
cd E:\mastercode\ultralytics-main-new
python 016_train_watermelon_seg_p2.py --model gpt_yaml_yolo\gpt_yolo11n_seg_p2_light_abc.yaml --name 17_gpt_light_abc
```

再跑精度版做主对比：

```powershell
python 016_train_watermelon_seg_p2.py --model gpt_yaml_yolo\gpt_yolo11n_seg_p2_psa_abcd.yaml --name 18_gpt_psa_abcd
```

## 接入二阶段关键点网络

训练完一阶段后，将新的 `best.pt` 传给 014/015：

```powershell
python 014_train_improved_v2.py --seg-model-path results\18_gpt_psa_abcd\weights\best.pt --seg-conf 0.25
python 015_train_distill_v2.py --seg-model-path results\18_gpt_psa_abcd\weights\best.pt --seg-conf 0.25
```

可视化 GT 与预测点：

```powershell
python 98_visualize_compare.py --image-path E:\mastercode\data\shr_watermelon\segmentation\images\val\dsc00005.jpg --seg-model-path results\18_gpt_psa_abcd\weights\best.pt
```

## 实验建议

记录 `mask mAP50`、`mask mAP50-95`、二阶段关键点 mAP、平均像素误差和推理速度。若精度版掩码更准但二阶段点位提升不明显，优先检查 YOLO 掩码与 GT 点匹配质量，而不是继续堆叠复杂模块。
