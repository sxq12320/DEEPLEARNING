# V5完整300轮结果与V6取舍

所有AP单位为%，AP50和框AP取最佳严格Mask AP同一轮。单seed42，不能称统计显著。

| 模型 | 峰值轮 | Mask AP50–95 | 同轮AP50 | 同轮Box AP50–95 | 尾20 Mask AP | 每轮秒中位数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SAGE30_official_control_seed42 | 184 | 66.826 | 82.652 | 70.492 | 66.160 | 14.79 |
| SAGE42_asym_semantic_detail_seed42 | 262 | 67.147 | 82.311 | 71.064 | 66.241 | 14.30 |
| SAGE50_late_proto_seed42 | 207 | 65.492 | 81.874 | 70.064 | 64.710 | 13.81 |
| SAGE51_detail_relay_seed42 | 161 | 67.218 | 82.610 | 70.670 | 66.235 | 13.34 |
| SAGE52_dual_route_seed42 | 177 | 65.497 | 82.307 | 69.918 | 64.469 | 219.30 |
| SAGE53_dual_boundary_seed42 | 193 | 65.832 | 82.679 | 70.707 | 64.952 | 15.39 |
| SAGE54_dual_neighbor_seed42 | 202 | 65.637 | 82.170 | 70.598 | 64.646 | 15.28 |
| SAGE55_dual_geometry_seed42 | 261 | 66.102 | 82.361 | 70.625 | 65.166 | 16.00 |
| SAGE56_dual_wt_p5_seed42 | 225 | 65.711 | 82.790 | 70.722 | 64.689 | 13.12 |

## 配方2×2消融

- late_proto_without_relay: -1.655个百分点。
- late_proto_with_relay: -1.721个百分点。
- relay_without_late_proto: +0.071个百分点。
- relay_with_late_proto: +0.005个百分点。

## 统一LOCAL诊断（不替代服务器CSV）

| 权重 | tiny n | tiny R@.001 | tiny R@.25 | 全体分裂代理 | 全体合并代理 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SAGE30_official_control_seed42_diagnostics | 153 | 46.41% | 16.99% | 2.10% | 3.72% |
| SAGE51_detail_relay_seed42_diagnostics | 153 | 46.41% | 16.34% | 1.72% | 2.67% |
| SAGE52_dual_route_seed42_diagnostics | 153 | 49.67% | 22.22% | 2.29% | 3.24% |

tiny为640输入stride4掩膜面积<256的分组；这里是Recall，不是AP_small。身份错误是需人工复核的代理。

结论：撤回默认late_proto；relay只有很弱的严格AP收益，未证明救回极小果。保留原型高分辨率细化，
把主干和颈部的预算重新分配作为下一轮结构假设。SAGE52的219秒级每轮耗时不应直接解释为该架构固有速度。

同批保存训练参数一致，676训练图/193验证图/1049验证实例。源码记录仅RUN_SAGE_V5.py与本地不一致；
本地入口还为screen/50，而实际args证实all/300。实现核心源文件在新增V6代码前已核对一致。

本次未重新进行全历史评估；历史对照边界沿用09-05重审报告，不与不同数据/AMP记录混排。
