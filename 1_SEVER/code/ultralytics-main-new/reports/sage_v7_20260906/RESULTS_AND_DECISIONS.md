# V6结果与V7实验依据

2026-09-06。所有服务器AP为百分数；AP50取严格Mask AP最佳同一轮。

| 模型 | 已有轮数 | Mask AP50–95 | 同轮AP50 | 前118轮最佳严格AP | 尾20严格AP | 每轮秒中位数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SAGE60_relay_control_seed42 | 300 | 67.503 | 82.603 | 66.977 | 66.994 | 19.88 |
| SAGE61_singlepass_neck_seed42 | 300 | 65.711 | 81.250 | 64.869 | 64.999 | 17.05 |
| SAGE62_residual_backbone_seed42 | 118 | 60.526 | 78.107 | 60.526 | 59.810 | 219.70 |
| SAGE63_persistent_detail_seed42 | 300 | 64.331 | 80.527 | 61.858 | 63.539 | 17.48 |
| SAGE64_selective_exchange_seed42 | 300 | 63.766 | 80.303 | 61.037 | 63.047 | 17.83 |

SAGE62只有118轮且无completed标记，不是完成的300轮结果；其约219.7秒每轮不是受控测速。
其余四组300轮并有完成标记；组内保存参数除模型和输出标识外一致。
新增V7前，V6记录的实现源码及初始化权重hash全部与本地相同。

历史覆盖：168份CSV，154份不同内容，262份改动前YAML配置。副本不算重复实验。详细对应见history_mapping。
读取范围是全CSV/配置盘点与关键模块实现复核，不声称逐行读完全部依赖；历史对应缺源码快照时仅为候选。

## 本地统一诊断（不替换服务器指标）

| 权重 | 极小实例数 | Mask R@.001 | Mask R@.25 | 较大组R@.25 |
| --- | ---: | ---: | ---: | ---: |
| SAGE60_relay_control_seed42_diagnostics | 153 | 43.14% | 15.03% | 96.60% |
| SAGE64_selective_exchange_seed42_diagnostics | 153 | 49.67% | 16.99% | 95.62% |

极小按640输入stride4栅格面积<256分组，是Recall而非COCO AP_small。
CPU FP32 batch1、best_mask、rectFalse；文件名核对不等于跨机器图像字节已核验。
SAGE60本地严格Mask AP约69.36%，高于服务器CSV的67.503%；此差异尚未归因，禁止用本地值替换或混排。

## SAGE60逐GT预测阶段诊断

- all，n=1049：{'success': 800, 'raw_box_present_score_low': 195, 'nms_limit_or_gt_competition': 5, 'no_raw_box_iou50': 44, 'matched_box_bad_mask': 5}
- tiny，n=153：{'success': 23, 'raw_box_present_score_low': 85, 'no_raw_box_iou50': 42, 'matched_box_bad_mask': 3}

固定conf=.25；先按框贪心一对一配对，再检查同一预测ID的mask。
NMS桶同时包含max_det与GT竞争，不是纯NMS因果分析；未分解实际训练TAL正样本质量。
极小实例：85个原始框存在但分数低、42个原始框IoU不足、3个匹配框的掩膜失败、23个成功。
这支持优先测试高分辨率候选表示/分类定位，但不证明P2必然有效，也不证明掩膜问题不存在。

## V7本地实测成本

| 模型 | 参数M | GFLOPs640 | 前向ms | 前向+损失+反向ms |
| --- | ---: | ---: | ---: | ---: |
| SAGE70_relay_control | 2.323 | 10.097 | 148.64 | 482.77 |
| SAGE71_compact_candidates | 1.784 | 8.040 | 105.21 | 479.58 |
| SAGE72_p2_candidates | 1.797 | 8.713 | 93.61 | 399.34 |
| SAGE73_p2_no_relay | 1.793 | 8.658 | 100.06 | 536.20 |
| SAGE74_p2_local_context | 1.790 | 8.350 | 93.92 | 604.79 |

CPU、FP32、batch1、640，20次合成微测；不含数据加载/优化器/NMS/验证，不是服务器FPS。
P2候选数从8400升至34000，TAL/分类与NMS成本仍须同GPU实测，低GFLOPs不保证低训练时间。

### 控制顺序波动后的交错测速（优先采用）

| 模型 | 前向ms | 前向+损失+反向ms |
| --- | ---: | ---: |
| SAGE70_relay_control | 103.98 | 482.15 |
| SAGE71_compact_candidates | 93.54 | 407.67 |
| SAGE72_p2_candidates | 100.81 | 479.77 |
| SAGE73_p2_no_relay | 104.48 | 454.43 |
| SAGE74_p2_local_context | 102.66 | 490.87 |

每轮随机模型顺序，3次预热+20次测量；CPU batch1 FP32 640。
71比70训练步约少15.4%，72/74与70接近。首次按模型连续测量的差异较大，不能据此断言模块速度。
无GPU速度或正式精度保证；测试环境Python3.9.13/Torch2.8.0 CPU。
