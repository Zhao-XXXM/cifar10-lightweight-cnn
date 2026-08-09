# 退化问题
- 网络并非越深越好，当网络加深到一定程度后，准确率可能不升反降
- 退化问题不是普通过拟合，而是深层网络本身变得难以优化
- 表现上常见为 Train Loss 和 Val Loss 同时较高，说明模型连训练集都学不好

## 网络加深导致性能变差，是因为过拟合吗？
- 不是
- 过拟合：Train Loss 很低，但 Val Loss 较高，说明训练集拟合得很好但泛化差
- 网络退化：Train Loss 和 Val Loss 都较高，说明模型优化困难，训练集也没学好

# 残差学习
- ResNet 的核心思想是让网络学习残差，而不是直接学习完整映射
- 传统网络希望若干层直接拟合目标函数 `H(x)`
- 残差网络将其改写为：`H(x) = F(x) + x`
- 其中 `F(x) = H(x) - x`，也就是残差映射

# 为什么残差连接有效
- 如果深层网络暂时学不到复杂映射，至少可以通过 shortcut 保留输入信息
- 当最优映射接近恒等映射时，学习 `F(x) = 0` 比直接学习 `H(x) = x` 更容易
- 残差连接为梯度提供更直接的传播路径，缓解深层网络训练困难

# BasicBlock
- BasicBlock 是 ResNet-18/34 使用的基础残差块
- 主分支结构为：`3 * 3 Conv -> BN -> ReLU -> 3 * 3 Conv -> BN`
- 最后将主分支输出与 shortcut 分支相加，再经过 ReLU
- `expansion = 1` 表示输出通道数不额外扩张

# Shortcut 分支
- 如果输入输出尺寸和通道数一致，shortcut 可以直接使用恒等映射
- 如果 stride 不为 1，或者输入输出通道数不同，就不能直接相加
- 此时需要使用 `1 * 1 Conv + BN` 对 shortcut 分支做维度对齐

# 两个测试场景
- 场景 1：`in_channels=64, out_channels=64, stride=1`，输入输出维度一致，可以直接相加
- 场景 2：`in_channels=64, out_channels=128, stride=2`，通道数翻倍且特征图降采样，需要 projection shortcut
- 这两个场景验证了残差块中维度对齐的重要性

# ResNet 中的两种基础残差块
- BasicBlock：用于 ResNet-18/34，由两个 3 * 3 卷积组成
- Bottleneck：用于 ResNet-50/101/152，由 1 * 1、3 * 3、1 * 1 卷积组成

# 今日自查问题
- 为什么残差块中主分支和 shortcut 分支必须形状一致？
- `stride=2` 通常会对特征图尺寸产生什么影响？
- BasicBlock 最后为什么是先相加再 ReLU？
