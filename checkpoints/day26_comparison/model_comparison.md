# 三模型对照实验

| 模型 | 参数量 | 参数量(M) | Checkpoint(MB) | Val Acc | 最佳 Epoch | 最终 Val Acc | 数据来源 | Acc/百万参数 |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| VGG-Slim | 815,018 | 0.8150 | 3.13 | 83.15% | not_recorded | 83.15% | checkpoint_eval | 102.02 |
| LightVGG-Slim | 563,403 | 0.5634 | 2.18 | 72.18% | 10 | 72.18% | history_json | 128.11 |
| LightVGG-Slim-GAP | 37,579 | 0.0376 | 0.17 | 74.88% | 10 | 74.88% | history_json | 1992.60 |

## 相对 VGG-Slim 的变化

| 模型 | 参数减少比例 | 最佳准确率差值 |
|---|---:|---:|
| VGG-Slim | 0.00% | +0.00 个百分点 |
| LightVGG-Slim | 30.87% | -10.97 个百分点 |
| LightVGG-Slim-GAP | 95.39% | -8.27 个百分点 |

> VGG-Slim 缺少 history JSON，因此表中 VGG-Slim 的 Val Acc 来自已保存 checkpoint 的重新评估；最佳 Epoch 与训练时间不补估计值。
