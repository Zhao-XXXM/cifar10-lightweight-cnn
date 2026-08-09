# Dropout
- Dropout 是一种正则化方法，用于缓解过拟合
- 训练时以概率 p 随机将部分神经元输出置 0
- 测试时使用完整网络进行稳定预测

# Inverted Dropout
- PyTorch 使用的是 Inverted Dropout
- 训练时被保留的神经元输出会除以 `1 - p`，保证输出期望基本不变
- 因此评估阶段不需要再手动缩放输出，`model.eval()` 会自动关闭 Dropout

# 为什么 Dropout 能防止过拟合
- 破除神经元之间的共适应性，避免某些神经元过度依赖另一些神经元
- 迫使每个神经元学习更独立、更稳健的特征
- 每次前向传播都相当于训练一个不同的子网络，具有类似集成学习的效果

# 代码中的 Dropout 放置位置
- Day9 的模型结构是 `SimpleCNN + BN + Dropout`
- Dropout 被放在展平后的全连接层之前
- 即先提取卷积特征，再对分类头输入做随机失活

# 为什么早期 Dropout 常放在全连接层
- 全连接层参数量通常较大，是过拟合高发位置
- 卷积层本身有局部连接和权值共享，参数量相对较少
- 图像特征图有空间相关性，直接在浅层卷积特征上做普通 Dropout 可能破坏局部结构

# train 和 eval 中 Dropout 的区别
- `model.train()` 模式下，Dropout 会随机失活神经元
- `model.eval()` 模式下，Dropout 会自动关闭
- 所以训练和评估前切换模式非常重要

# 实验现象
- 单 Epoch 下，加入 Dropout 后准确率可能比 Day8 略低
- 原因是 Dropout 在训练时增加了学习难度，短期内像给模型加了负重
- 多 Epoch 训练时，它更有机会体现防止过拟合、提升泛化的价值
- 梳理图：![dropout summary](image/image-7.png)
- 训练效果：![dropout result](image/image-6.png)

# 今日自查问题
- `Dropout(p=0.5)` 中 p 表示保留概率还是丢弃概率？
- 为什么评估阶段必须关闭 Dropout？
- Dropout 是减少参数量，还是改变训练时的特征使用方式？
