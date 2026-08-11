# LightVGG-Slim 第一次训练
- Day21 开始训练 Day20 搭建的 LightVGG-Slim
- 今天的核心目标是观察：在训练配置保持一致时，轻量卷积是否能保住原始 VGG-Slim 的分类效果
- 这是一组结构对照实验，不是单纯追求某一个模型的最高准确率

# 公平对照实验
- VGG-Slim 和 LightVGG-Slim 使用相同的 CIFAR-10 数据集
- 使用相同的标准化参数
- 使用相同的 Batch Size、学习率、优化器和 Epoch 数
- 暂时不加入数据增强、Weight Decay 或学习率调度，避免一次改变太多变量
- 当前唯一主要变量：普通 3 * 3 卷积是否替换为 Depthwise Separable Conv

# 训练指标
- Train Loss：训练集上的平均交叉熵损失
- Train Acc：训练集上的分类准确率
- Val Loss：测试集上的平均交叉熵损失
- Val Acc：测试集上的分类准确率
- 训练集指标看拟合能力，验证集指标看泛化能力

# 为什么要按样本数计算平均 Loss
- 每个 Batch 的最后一批样本数可能不足 Batch Size
- 如果直接平均每个 Batch 的 loss，会让小 Batch 和完整 Batch 权重相同
- 代码使用 `loss.item() * images.size(0)` 累计总损失，最后除以总样本数
- 这样得到的是严格的样本平均损失

# 训练与评估模式
- `model.train()` 让 Dropout 和 BatchNorm 按训练模式工作
- `model.eval()` 让 Dropout 关闭，并让 BatchNorm 使用运行统计量
- 验证阶段使用 `torch.no_grad()`，避免构建不必要的计算图

# 最佳权重保存
- 每个 Epoch 验证后比较当前 `Val Acc` 和历史最高准确率
- 只保存验证集准确率最高的权重
- 当前 checkpoint 路径为 `checkpoints/light_vgg_slim_best.pth`
- 同时保存 `light_vgg_slim_history.json`，便于后续绘制曲线和整理实验表格

# 运行方式
- 冒烟测试：`venv\Scripts\python.exe notecode\d21_train_light_cnn.py --epochs 1`
- 正式训练：`venv\Scripts\python.exe notecode\d21_train_light_cnn.py --epochs 10`
- 先确认 1 Epoch 能完整跑通，再进行正式实验

# 结果分析方向
- 如果 LightVGG-Slim 准确率接近 VGG-Slim，说明轻量卷积保留了较好的表达能力
- 如果准确率明显下降，说明拆分卷积可能损失了部分通道交互能力
- 不能只看准确率，还要同时记录参数量、模型大小和训练时间

# 今日自查问题
- 为什么 Day21 暂时不加入数据增强和 Cosine Scheduler？
- 为什么验证阶段要调用 `model.eval()` 和 `torch.no_grad()`？
- 如果训练集准确率很高、验证集准确率明显较低，说明什么？
- 轻量模型准确率下降多少时，你认为需要继续优化？
