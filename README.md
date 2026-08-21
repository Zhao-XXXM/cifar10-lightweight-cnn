# CIFAR-10 Lightweight CNN

一个面向学习和科研实践的 CIFAR-10 图像分类项目。项目从普通 CNN 出发，逐步实现深度可分离卷积、全局平均池化、宽度系数实验、多随机种子评估、理论复杂度分析、真实 CPU 推理基准和 SE 注意力消融。

## 项目目标

- 理解卷积神经网络的前向传播、损失函数和反向传播
- 分析深度可分离卷积和 GAP 对参数量、FLOPs 和表达能力的影响
- 使用公平实验比较不同模型宽度
- 通过多随机种子和配对消融减少偶然性
- 建立从模型设计到最终测试评估的规范科研流程

## 技术栈

- Python
- PyTorch / torchvision
- NumPy
- Matplotlib
- CIFAR-10

## 模型结构

最终候选模型是 `WidthLightVGGSlimGAP(width_mult=1.0)`：

```text
Input 3x32x32
  -> Depthwise Separable Conv x2 -> MaxPool
  -> Depthwise Separable Conv x2 -> MaxPool
  -> Depthwise Separable Conv x2 -> MaxPool
  -> Global Average Pooling
  -> Linear classifier
```

深度可分离卷积由两步组成：

1. Depthwise convolution：每个输入通道独立进行空间卷积。
2. Pointwise convolution：使用 1x1 卷积完成通道混合。

与普通卷积相比，它通常能减少计算量，但也可能削弱通道间的联合建模能力，因此必须通过实验验证准确率变化。

## 环境部署

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

当前开发环境使用 Python 3.13.5、PyTorch 2.13.0+cpu、torchvision 0.28.0+cpu、NumPy 2.5.1、Matplotlib 3.11.0 和 Pillow 12.3.0。依赖下限见 `requirements.txt`。

下载并解压 CIFAR-10 Python 版本，使目录结构如下：

```text
data\cifar-10-batches-py\
  data_batch_1
  data_batch_2
  data_batch_3
  data_batch_4
  data_batch_5
  test_batch
  batches.meta
```

本次开发机曾使用 `data_day38` 作为等内容的数据目录，原因是原解压目录存在 Windows 文件权限问题。标准目录仍然是 `data/`；当前开发机如遇同样权限问题，可在命令中增加 `--data-dir data_day38`。两个数据目录都不应提交到 Git 仓库。

## 运行项目

各天脚本位于 `notecode/`，对应学习笔记位于 `note/`。核心正式流程是 Day31-Day40；Day01-Day30 主要用于逐步学习和历史探索。

Day28-Day30 的部分脚本曾直接读取官方 Test Set 作为探索性评估，相关结果不属于最终正式结果，不能与 Day32 之后的验证集结果混用。正式复现实验从 Day31 的数据划分开始。

```powershell
# Day39：汇总已有实验并选择最终候选
venv\Scripts\python.exe notecode\d39_model_selection.py

# Day40：最终候选的官方 Test Set 评估
venv\Scripts\python.exe notecode\d40_final_test.py --data-dir data_day38
```

运行模型结构冒烟测试：

```powershell
venv\Scripts\python.exe -m unittest discover -s tests -v
```

最终评估默认使用 `data`；本次开发机运行时显式传入了 `data_day38`。评估对象固定为：

- 数据目录：默认 `data`；本次开发机使用 `data_day38`
- 模型：`width_mult=1.0`
- checkpoint：Day33 的 seed=42、123、2026
- Batch Size：128
- 设备：自动选择 CUDA，否则使用 CPU

## 实验协议

- 官方 Train：45,000 张训练图像 + 5,000 张验证图像
- 数据划分随机种子：42
- 划分 SHA-256：`6242d545f1d70bbd004ba26cd92784461728e0ffb0a64a1f27d1a6421039967e`
- 正式训练种子：42、123、2026
- 优化器：Adam
- 学习率：0.001
- Batch Size：64
- Epoch：10
- 损失函数：CrossEntropyLoss
- 最佳 checkpoint：验证集准确率最高的 epoch
- 官方 Test Set：模型选择完成后进入一次最终评估阶段；该阶段评估预先固定的三个 seed checkpoint，不再据结果修改模型

## 实验结果

### 宽度与成本

| 模型 | 验证准确率 | 参数量 | FLOPs/图像 | Batch=1 中位延迟 |
| --- | ---: | ---: | ---: | ---: |
| width=0.5 | 67.56%* | 10,875 | 2.879M | 1.566 ms |
| width=1.0 | 73.84% +/- 1.30% | 37,579 | 9.896M | 1.977 ms |
| width=1.5 | 74.72%* | 80,155 | 21.108M | 2.521 ms |

`*` 表示目前只有 seed=42 的单次探索结果。width=1.0 是唯一完成三种子稳定性验证的宽度。

### SE 消融

| 模型 | Best Val Acc 均值 | 样本标准差 | 参数量 | 平均训练时间 |
| --- | ---: | ---: | ---: | ---: |
| Baseline width=1.0 | 73.84% | 1.30% | 37,579 | 898.48 s |
| Baseline + SE | 73.49% | 1.50% | 43,431 | 1062.68 s |

SE 平均准确率下降 0.35 个百分点，参数量增加 15.57%，平均训练时间增加约 18.27%。因此没有采用 SE。

### 最终官方测试集

最终候选固定后，在 10,000 张官方 Test Set 上进行推理：

| 训练 seed | Test Accuracy | Test Loss |
| ---: | ---: | ---: |
| 42 | 71.26% | 0.8276 |
| 123 | 74.10% | 0.7547 |
| 2026 | 72.71% | 0.8021 |
| 均值 +/- 样本标准差 | 72.69% +/- 1.42% | - |

测试集只用于最终报告，没有参与模型、超参数或 checkpoint 选择。

## 类别分析

三个最终 checkpoint 的测试预测聚合结果：

| 类别 | 准确率 |
| --- | ---: |
| automobile | 86.50% |
| ship | 86.20% |
| truck | 80.50% |
| horse | 77.77% |
| airplane | 75.90% |
| frog | 75.47% |
| deer | 72.77% |
| dog | 59.90% |
| cat | 58.23% |
| bird | 53.67% |

错误主要集中在外观相似类别，例如 cat/dog、bird/deer 和 airplane/ship。CIFAR-10 的 32x32 低分辨率也限制了细粒度辨别能力。

## 目录结构

```text
notecode/                 # 按天拆分的实验脚本
note/                     # 按天学习笔记
tests/                    # 不读取数据的快速结构测试
checkpoints/              # history、CSV、图像和模型 checkpoint
data/                     # CIFAR-10 数据，不提交
data_day38/               # 当前可读的数据目录，不提交
```

## 主要产物

- `checkpoints/day39_model_selection/model_summary.csv`
- `checkpoints/day39_model_selection/accuracy_cost_pareto.png`
- `checkpoints/day40_final_evaluation/test_summary.csv`
- `checkpoints/day40_final_evaluation/class_accuracy.csv`
- `checkpoints/day40_final_evaluation/confusion_matrix.png`
- `checkpoints/day40_final_evaluation/final_report.json`

## 项目结论

在当前 CIFAR-10、训练预算和 CPU 部署约束下，`width_mult=1.0` 是准确率、复杂度、推理速度和实验可靠性之间更均衡的方案。这个项目的重点不是宣称提出了新的 SOTA 模型，而是完整展示了一个可复现的轻量 CNN 研究流程：提出假设、控制变量、设计对照、量化成本、分析负结果、固定模型后再进行最终测试。

## 局限与后续方向

- width=0.5 和 width=1.5 还需要补充三种子实验，才能做严格稳定性比较
- 当前只测试了 CPU 单线程推理，尚未覆盖 ARM、GPU 或移动端 NPU
- 可以加入学习率调度、标签平滑、MixUp、CutMix 等训练策略并设计独立对照
- 可以尝试蒸馏、结构化剪枝、量化和 ONNX/TorchScript 部署
- 若研究重点转向论文，需要扩大数据集或任务，并与更多轻量模型基线比较

## 学习记录

完整的 Day01-Day40 学习和实验过程见 `note/`。
