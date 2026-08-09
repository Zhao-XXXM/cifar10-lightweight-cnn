# 训练模型
- 训练就是给模型输入大量数据和对应标签，让模型通过反向传播自动调整参数
- 数学本质：拟合一个复杂函数 `f(x; W, b)`，让它对输入图片 x 输出尽可能接近真实标签的结果
- 对图像分类来说，模型学习到的不是固定像素，而是边缘、纹理、形状等可泛化视觉特征

# 数据预处理
- `transforms.ToTensor()` 将图片转成 Tensor，并把像素值缩放到 `[0, 1]`
- `Normalize(mean, std)` 对每个通道做标准化，使数据分布更稳定
- CIFAR-10 常用均值为 `(0.4914, 0.4822, 0.4465)`，标准差为 `(0.2023, 0.1994, 0.2010)`

# DataLoader
- `DataLoader` 负责把 Dataset 打包成一个个 Batch
- `batch_size=64` 表示每次训练读取 64 张图片
- `shuffle=True` 表示每个 Epoch 打乱训练集顺序，减少模型记住数据排列的风险

# SimpleCNN 训练结构
- 输入：`(64, 3, 32, 32)`
- 第一组卷积池化后：`(64, 16, 16, 16)`
- 第二组卷积池化后：`(64, 32, 8, 8)`
- 展平后：`(64, 2048)`
- 最终输出：`(64, 10)`，每一行对应一张图片的 10 类 logits

# CrossEntropyLoss
- `nn.CrossEntropyLoss()` 用于多分类任务
- 它内部已经包含 `LogSoftmax + NLLLoss`
- 因此模型最后一层直接输出 logits，不需要手动加 Softmax
- labels 应该是类别编号，如 0 到 9，而不是 one-hot 向量

# Adam 优化器
- Adam 会结合一阶动量和二阶动量，自适应调整每个参数的更新幅度
- 相比基础 SGD，Adam 在初学阶段更容易稳定收敛
- 代码中使用 `lr=0.001`，这是 Adam 常用的起始学习率

# 训练过程
- 前向传播：`outputs = model(images)`
- 计算损失：`loss = criterion(outputs, labels)`
- 梯度清零：`optimizer.zero_grad()`
- 反向传播：`loss.backward()`
- 参数更新：`optimizer.step()`
- 每隔 100 个 Batch 打印一次 loss，用于观察训练是否正常下降

# 神经网络在视觉任务中学到了什么
- 浅层卷积常学习边缘、颜色变化和简单纹理
- 中层卷积会组合出局部形状和复杂纹理
- 深层卷积逐渐学习类别相关的高级语义
- 视觉特征层次示意：![visual hierarchy](image/image-3.png)

# 泛化能力
- 过拟合：训练集准确率很高，但测试集准确率很低
- 泛化能力：模型学到通用规律，对没见过的新图片仍能正确分类
- 后续的 BatchNorm、Dropout、数据增强都是为了提升泛化能力

# 今日自查问题
- 为什么分类任务使用 CrossEntropyLoss，而不是 MSELoss？
- 为什么模型输出是 10 维 logits，而不是一个类别名？
- 为什么训练集要 `shuffle=True`？
