# Checkpoint（最佳模型权重）保存机制
- 在“同一次训练过程”中，在几十个 Epoch（轮次）里，实时捕捉并保存模型状态最好（泛化能力最强）的那一个时刻
- 防止训练崩溃丢权重：如果训练 100 个 Epoch，在第 99 个 Epoch 突然断电或报错，如果没有保存，前面的计算就白费了
- 解决“过拟合”陷阱：在多 Epoch 训练中，随着 Epoch 增加，train_loss 会持续下降，但 val_loss 可能会在某个点后开始上升（过拟合）。我们不能直接拿最后一个 Epoch 的模型，而是要在训练过程中，实时监测 Validation Accuracy，只保存验证集准确率历史最高的那一次模型权重

# PyTorch 权重保存的底层机制：state_dict
- 保存整个模型：torch.save(model, 'model.pth')：依赖固定的代码目录结构，跨文件加载容易报错
- 保存字典权重：torch.save(model.state_dict(), 'best_model.pth')：state_dict 本质上是一个 Python 字典（Key-Value），存储了网络中每一层可学习的张量参数