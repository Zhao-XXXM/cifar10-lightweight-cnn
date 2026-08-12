# 消融实验（Ablation Study）
- 消融实验的目的不是一次加入所有技巧，而是逐个移除或加入某个组件，观察它对结果的独立影响
- 它能回答“模型为什么变好”，比只报告一个最终准确率更有科研价值
- Day23 固定 LightVGG-Slim 主体结构，只改变训练策略

# 实验变量控制
- 模型结构：LightVGG-Slim
- 数据集：CIFAR-10
- Batch Size：64
- 初始学习率：0.001
- 优化器：Adam
- 训练轮数：10 Epoch
- 随机种子：42
- 每组实验只改变当前研究的训练策略，其他条件保持一致

# 四组实验设计
- Baseline：无数据增强、无 Weight Decay、无 Cosine Scheduler
- Augmentation：在 Baseline 上加入 `RandomCrop + RandomHorizontalFlip`
- Weight Decay：在 Augmentation 上加入 `weight_decay=5e-4`
- Cosine：在 Weight Decay 上加入 `CosineAnnealingLR`

# 为什么采用逐步加入策略
- `Baseline -> Augmentation` 可以观察数据增强的独立贡献
- `Augmentation -> Weight Decay` 可以观察 L2 正则化的额外贡献
- `Weight Decay -> Cosine` 可以观察学习率调度的额外贡献
- 如果一次同时打开全部策略，就无法判断是哪一种策略带来了性能变化

# 数据增强的作用
- `RandomCrop(32, padding=4)` 模拟图片发生轻微平移
- `RandomHorizontalFlip()` 增加左右方向变化
- 数据增强只用于训练集，测试集保持确定性的 `ToTensor + Normalize`
- 它改变的是输入样本分布，不改变模型参数量

# Weight Decay 的作用
- Weight Decay 通过惩罚过大的权重抑制过拟合
- 在 Adam 中通过 `weight_decay=5e-4` 传入
- 它不会改变模型结构和参数量，但会改变参数更新规则
- Weight Decay 过大可能限制模型拟合能力，过小则正则化效果不明显

# Cosine Scheduler 的作用
- Cosine Scheduler 让学习率随 Epoch 平滑下降
- 前期较大的学习率帮助快速搜索，后期较小的学习率帮助精细收敛
- 代码在每个 Epoch 结束后调用 `scheduler.step()`
- 当前实验中 `T_max=epochs`，表示整个训练过程完成一个余弦退火周期

# 指标记录
- 每组实验记录 Train Loss、Train Acc、Val Loss、Val Acc
- 额外记录最佳 Val Acc、最佳 Epoch 和训练耗时
- 结果汇总表至少包含：实验名、参数量、最佳准确率、最佳 Epoch、训练耗时

# 当前实验的局限
- 每组实验只使用一个随机种子，结果会受到随机初始化和 Batch 顺序影响
- 更严谨的研究应使用多个随机种子，报告平均值和标准差
- 当前阶段先用单种子理解变量影响，后续可以把多种子作为扩展实验

# 今日自查问题
- 为什么一次加入所有技巧不能算严格的消融实验？
- 数据增强会改变参数量吗？
- Weight Decay 和 Cosine Scheduler 分别改变了什么？
- 如果某个策略使准确率下降，是否一定说明它没有价值？
