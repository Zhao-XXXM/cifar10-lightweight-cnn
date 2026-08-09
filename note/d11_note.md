# Checkpoint 保存机制
- Checkpoint 用于在训练过程中保存模型权重
- Day11 中每个 Epoch 都会验证一次模型，如果验证集准确率超过历史最好值，就保存当前权重
- 这样可以捕捉同一次训练中泛化能力最好的模型状态

# 为什么不能只保存最后一个 Epoch
- 多 Epoch 训练时，train loss 可能持续下降，但 val loss 可能后期上升
- 这说明模型开始过拟合训练集，最后一个 Epoch 不一定是泛化最好的模型
- 因此应该监控验证集指标，保存 `best_acc` 对应的模型

# 防止训练中断损失
- 如果训练过程中断电或报错，checkpoint 可以保留已经训练出的权重
- 对更长时间训练尤其重要
- 这也是规范深度学习工程必须具备的能力

# state_dict
- `model.state_dict()` 本质是一个 Python 字典
- key 是每一层参数的名字，value 是对应的 Tensor 权重
- 推荐保存方式：`torch.save(model.state_dict(), "best_model.pth")`
- 不推荐初学阶段直接保存整个模型对象，因为它更依赖代码路径和类定义

# 设备选择
- `torch.device("cuda" if torch.cuda.is_available() else "cpu")` 会优先使用 GPU
- `model.to(device)` 把模型参数移动到对应设备
- 每个 Batch 中的 `images` 和 `labels` 也必须 `.to(device)`，否则会出现设备不一致报错

# Epoch 级别指标
- `running_loss += loss.item() * images.size(0)` 是为了按样本数累计 loss
- 最终 `epoch_train_loss = running_loss / total_train` 得到整个训练集的平均 loss
- Train Acc 和 Val Acc 分别用于观察拟合能力和泛化能力

# 保存目录
- `os.makedirs(save_dir, exist_ok=True)` 会在不存在时创建 `checkpoints` 文件夹
- `exist_ok=True` 表示文件夹已存在时不报错
- 最佳权重保存为 `checkpoints/vgg_slim_best.pth`

# 今日自查问题
- 为什么保存最佳模型要看验证集准确率，而不是训练集准确率？
- `state_dict` 保存的是模型结构还是模型参数？
- 为什么模型和数据必须放在同一个 device 上？
