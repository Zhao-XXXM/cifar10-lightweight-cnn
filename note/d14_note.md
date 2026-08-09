# ResNet-18 整体结构
- Day14 将 BasicBlock 组装成完整 ResNet-18
- 经典 ResNet-18 由 Stem、4 个 Layer、全局平均池化和全连接分类器组成
- 针对 CIFAR-10，Stem 使用 3 * 3 卷积，而不是 ImageNet 版本中的 7 * 7 大卷积

# Stem 层
- Stem 层负责将 RGB 输入图像转换为 64 通道特征图
- 代码结构：`Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) -> BN -> ReLU`
- CIFAR-10 图像只有 32x32，如果一开始就大步长降采样，会过早丢失细节

# 四个残差 Stage
- Layer 1：2 个 BasicBlock，输出通道数 64，不降采样
- Layer 2：2 个 BasicBlock，输出通道数 128，第一个 Block 使用 `stride=2` 降采样
- Layer 3：2 个 BasicBlock，输出通道数 256，第一个 Block 使用 `stride=2` 降采样
- Layer 4：2 个 BasicBlock，输出通道数 512，第一个 Block 使用 `stride=2` 降采样

# _make_layer 函数
- `_make_layer(out_channels, num_blocks, stride)` 用于批量构造一个 Stage
- `strides = [stride] + [1] * (num_blocks - 1)` 表示只有第一个 Block 负责降采样
- 每添加一个 Block 后，`self.in_channels` 会更新为当前 Stage 的输出通道数
- 这样后续 Block 才能正确接收上一层输出

# 全局平均池化
- `AdaptiveAvgPool2d((1, 1))` 会把任意空间尺寸的特征图压缩为 `1x1`
- 它减少了分类头中全连接层的参数量
- 与直接 Flatten 大特征图相比，全局平均池化更符合现代 CNN 设计

# 分类器
- 经过全局平均池化后，特征形状为 `(Batch, 512, 1, 1)`
- `torch.flatten(out, 1)` 从第 1 维开始展平，得到 `(Batch, 512)`
- 最后 `Linear(512, 10)` 输出 CIFAR-10 的 10 类 logits

# 形状验证
- 代码使用 `dummy_input = torch.randn(2, 3, 32, 32)` 模拟 CIFAR-10 输入
- 输出形状应该是 `(2, 10)`
- 这说明模型可以接收一个 Batch 的图片，并为每张图输出 10 个类别分数

# 今日自查问题
- 为什么 CIFAR-10 版本 ResNet 不适合开头使用 7 * 7 stride=2 卷积？
- `_make_layer` 中为什么只有第一个 Block 可能使用 `stride=2`？
- 全局平均池化相比大 Flatten 分类头有什么优势？
