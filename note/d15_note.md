# ResNet-18 训练
- Day15 将 Day14 的 ResNet-18 放到 CIFAR-10 上进行完整训练
- 训练流程与 VGG-Slim 保持一致：训练、验证、记录指标、保存最佳权重
- 这样可以公平比较不同模型结构在同一任务上的表现

# 与 VGG-Slim 的对比
- ResNet-18 参数量明显大于 VGG-Slim
- 但残差连接让更深的网络依然可以稳定训练
- 实验现象：虽然 ResNet-18 参数量约为 VGG-Slim 的十几倍，但训练过程仍然比较稳定，Train Loss 能持续下降

# 训练配置
- 数据集：CIFAR-10
- 预处理：`ToTensor + Normalize`
- Batch Size：64
- 损失函数：`CrossEntropyLoss`
- 优化器：`Adam(lr=0.001)`
- 训练轮数：10 Epoch

# 指标记录
- 每个 Epoch 统计 Train Loss、Train Acc、Val Loss、Val Acc
- Train 指标反映模型对训练集的拟合能力
- Val 指标反映模型对未见数据的泛化能力
- 最终保存验证集准确率最高的权重，而不是保存最后一轮权重

# checkpoint
- 最佳权重保存路径：`checkpoints/resnet18_best.pth`
- 保存逻辑仍然是 `if val_acc > best_acc`
- 这说明 VGG-Slim 和 ResNet18 已经具备统一实验流程，后续才能做规范对照实验

# 为什么 ResNet18 能训练更深
- BasicBlock 中的 shortcut 分支为信息和梯度提供了更直接的通路
- 深层网络不用每一层都从零学习完整映射，而是学习残差
- 这缓解了深层网络退化问题

# 今日自查问题
- 为什么比较 VGG-Slim 和 ResNet18 时要尽量保持训练配置一致？
- ResNet18 参数量更大，是否一定泛化更好？
- 为什么最佳模型要根据 Val Acc 保存？
