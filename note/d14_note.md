#  经典的ResNet-18 由以下几个部分顺次连接组成
- Stem 层（输入预处理层）：针对 CIFAR-10，用一个  卷积把 RGB 图像转为 64 通道
- Layer 1：2 个 BasicBlock（通道数 64，不降采样）
- Layer 2：2 个 BasicBlock（通道数 128，首个 Block 降采样 stride=2）
- Layer 3：2 个 BasicBlock（通道数 256，首个 Block 降采样 stride=2）
- Layer 4：2 个 BasicBlock（通道数 512，首个 Block 降采样 stride=2）
- Classifier（分类器）：全局平均池化（AdaptiveAvgPool2d）  全连接层（Linear）