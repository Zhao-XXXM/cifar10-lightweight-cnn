# Bottleneck 残差块
- Bottleneck 是 ResNet-50/101/152 中使用的残差块结构，相比 BasicBlock 更适合构建更深的网络
- BasicBlock 由两个 3 * 3 卷积组成；Bottleneck 由 1 * 1、3 * 3、1 * 1 三个卷积组成
- 它的核心思想是：先用 1 * 1 卷积降低通道数，再用 3 * 3 卷积提取空间特征，最后再用 1 * 1 卷积恢复或扩展通道数

# 为什么叫 Bottleneck
- Bottleneck 的含义是“瓶颈”，指中间通道数被压缩，形成一个较窄的特征通道
- 假设输入通道数为 in_channels，中间通道数为 mid_channels，输出通道数通常为 mid_channels * 4
- 这种设计能减少 3 * 3 卷积直接在高通道特征上计算带来的参数量和计算量

# 三层卷积的作用
- 第一个 1 * 1 卷积：负责降维，将输入通道数压缩到 mid_channels
- 中间 3 * 3 卷积：负责真正提取局部空间特征，如果 stride=2，还会完成特征图降采样
- 最后一个 1 * 1 卷积：负责升维，将通道数扩展为 mid_channels * 4

# Bottleneck 与残差连接
- Bottleneck 仍然满足 ResNet 的核心公式：H(x) = F(x) + x
- 主分支 F(x) 由三层卷积组成，shortcut 分支用于保留原始输入信息
- 如果输入输出的尺寸或通道数不一致，就需要在 shortcut 分支使用 1 * 1 卷积进行维度对齐

# Bottleneck 的参数量优势
- 如果直接使用高通道的 3 * 3 卷积，参数量会随着输入通道和输出通道快速增加
- Bottleneck 先降维再做 3 * 3 卷积，可以让最耗参数的空间卷积发生在较低通道数上
- 这说明 1 * 1 卷积不仅能调整通道数，也是一种重要的轻量化设计思想

# 与后续轻量化路线的关系
- Bottleneck 让我们第一次看到：减少参数量不一定只靠减少层数，也可以通过重新设计卷积结构实现
- 后续的 Depthwise Separable Convolution 也是类似思路：把普通卷积拆成更便宜的计算步骤
- 因此 Bottleneck 是从 ResNet 过渡到轻量化 CNN 的重要桥梁
