# 模型复杂度统计
- 轻量化模型不能只看准确率，还要同时看参数量、模型文件大小、训练时间和推理速度
- 参数量越大，模型通常需要更多存储空间，也更容易带来更高计算开销
- Day18 的目标是学会用公式和代码统计模型参数量，为后续轻量化实验提供量化依据

# 卷积层参数量公式
- 普通二维卷积的权重形状为：out_channels * in_channels * kernel_size * kernel_size
- 如果卷积层使用 bias，则还要额外加上 out_channels 个偏置参数
- 因此普通卷积参数量为：out_channels * in_channels * k * k + bias

# 例子
- 对于 Conv2d(3, 32, kernel_size=3)，如果有 bias，参数量为：32 * 3 * 3 * 3 + 32 = 896
- 对于 Conv2d(64, 128, kernel_size=3, bias=False)，参数量为：128 * 64 * 3 * 3 = 73728
- 可以看到，当输入通道数和输出通道数变大时，3 * 3 卷积的参数量会快速增加

# BatchNorm 参数量
- BatchNorm2d(C) 中可训练参数主要是 gamma 和 beta
- 每个通道各有一个 gamma 和一个 beta，因此可训练参数量为：2 * C
- running_mean 和 running_var 会被保存，但通常不算作可训练参数

# 全连接层参数量
- Linear(in_features, out_features) 的权重形状为：out_features * in_features
- 如果有 bias，则再加 out_features
- 因此全连接层参数量为：out_features * in_features + out_features

# 参数量与模型大小
- PyTorch 默认通常使用 float32 保存参数，每个参数约占 4 字节
- 估算模型大小可以用：参数量 * 4 / 1024 / 1024 MB
- 实际 checkpoint 文件大小可能略有差异，因为还可能保存 BatchNorm 的统计量和文件结构信息

# 为什么这是轻量化的第一步
- 只有先统计基线模型的参数量，后面才能证明轻量化模型是否真的变小
- VGG-Slim、ResNet18、Bottleneck 的参数量差异，能帮助我们理解不同结构设计的代价
- 后续 Depthwise Separable Convolution 的核心价值，也要通过参数量对比体现出来

# 今日自查问题
- 普通卷积的参数量为什么和输入通道数、输出通道数都有关？
- BatchNorm 为什么会带来少量可训练参数？
- 参数量少是否一定代表准确率更高？
- 参数量少是否一定代表推理速度更快？

# 为什么 layer4 比前几层参数多很多
- ![alt text](image.png)
- 普通卷积的参数量同时受输入通道数、输出通道数、卷积核大小影响