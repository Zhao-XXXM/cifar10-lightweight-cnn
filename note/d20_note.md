# Light VGG-Slim
- Day20 的目标是把 Day19 学到的 Depthwise Separable Convolution 放进完整 CNN 模型
- 我们先不追求训练准确率，而是验证轻量模型的结构、输出形状和参数量
- 为了公平对比，整体网络仍然沿用 Day10 的 VGG-Slim 三段式结构

# 从 VGGBlock 到 LightVGGBlock
- 原始 VGGBlock 中，每个 3 * 3 普通卷积同时完成空间特征提取和通道融合
- LightVGGBlock 中，每个普通卷积被替换为 Depthwise Separable Conv
- 只替换卷积结构，先保持整体网络不变，用来控制变量
- 也就是：`Depthwise 3 * 3 -> Pointwise 1 * 1`

# 为什么先保持整体结构不变
- 轻量化实验要尽量控制变量
- 如果同时改卷积、改通道数、改分类头，就很难判断准确率或参数量变化来自哪一个改动
- Day20 只替换卷积结构，后面再逐步加入全局平均池化、通道缩放等改进

# LightVGGBlock 结构
- 原始 VGGBlock：`Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> MaxPool`
- LightVGGBlock：`DSConv -> DSConv -> MaxPool`
- 每个 DSConv 内部包含：`Depthwise Conv -> BN -> ReLU -> Pointwise Conv -> BN -> ReLU`

# 输入输出尺寸变化
- 输入图片：`3x32x32`
- 第 1 个 LightVGGBlock：`3x32x32 -> 32x16x16`
- 第 2 个 LightVGGBlock：`32x16x16 -> 64x8x8`
- 第 3 个 LightVGGBlock：`64x8x8 -> 128x4x4`
- 分类头仍然接收 `128 * 4 * 4 = 2048` 维特征

# 参数量变化的关键
- 普通 3 * 3 卷积参数量：`Cin * Cout * 3 * 3`
- 深度可分离卷积参数量：`Cin * 3 * 3 + Cin * Cout`
- 通道数越大，替换后的参数节省越明显

# 一个重要观察
- 如果只替换卷积层，而分类头保持 `Linear(2048, 256)`，模型中仍然会有不少参数集中在全连接层
- 所以轻量化不只包括卷积轻量化，还包括分类头设计、通道数设计和整体结构设计
- 这为后续使用 Global Average Pooling 做进一步轻量化埋下伏笔

# 今日自查问题
- 为什么 Day20 先保持 VGG-Slim 的整体结构不变？
- LightVGGBlock 和 VGGBlock 的输出尺寸为什么可以保持一致？
- 如果轻量卷积已经减少了很多参数，为什么全连接层仍可能成为参数大户？
- 今天的 LightVGG-Slim 属于最终模型，还是轻量化路线中的第一个版本？
