# Day32：在规范 Train/Validation 划分上重训宽度模型
- Day31 已经把 CIFAR-10 官方训练集分成 45,000 张 Train 和 5,000 张 Validation
- Day32 使用同一份 `seed=42` 索引重新训练 `width_mult=0.5、1.0、1.5`
- 三个模型只通过 Validation Acc 选择最佳 Epoch，今天不读取官方 Test Set
- 目标是建立可以用于后续多随机种子和复杂度分析的规范基线

# 训练、验证和测试的完整流程
- 每个 Batch 的 Train 数据参与前向传播、损失计算、反向传播和参数更新
- 每个 Epoch 结束后在 Validation Set 上推理，但不执行反向传播
- 如果当前 Val Acc 高于历史最好值，就保存当前 checkpoint
- 三个宽度模型全部训练完成后，根据 Validation 结果和资源成本形成候选模型
- 官方 Test Set 暂时保持不读取，最终方案确定后再进行正式测试

# 从经验风险理解数据划分
- 训练过程最小化训练集上的平均损失：`R_train(theta) = 1/N * sum L(f_theta(x_i), y_i)`
- 模型参数 `theta` 由 Train Set 学习得到
- 模型结构、宽度和最佳 Epoch 属于模型选择，也需要独立的 Validation Set
- Test Set 用来估计最终方案对未知数据的泛化能力，不应该反复反馈到设计过程

# `Subset` 的作用
- CIFAR-10 官方训练集仍然是一个包含 50,000 张图的 Dataset
- `torch.utils.data.Subset(dataset, indices)` 只暴露索引列表指定的样本
- Train Subset 使用 Day31 的 45,000 个训练索引
- Validation Subset 使用 Day31 的 5,000 个验证索引
- `validate_split` 会再次检查两个索引集合无重复、无交集并完整覆盖官方训练集

# 为什么创建两个 Dataset 实例
- 当前 Train 和 Validation 都只使用 ToTensor 与 Normalize，预处理暂时相同
- 代码仍然分别创建 `train_source` 和 `val_source`
- 后续如果训练集加入 RandomCrop 或 RandomHorizontalFlip，只修改训练变换即可
- Validation 必须保持确定性，不能使用随机数据增强，否则同一模型每次评估结果可能变化

# 公平对照实验设置
- 数据划分：Day31 `seed=42` 分层划分
- 训练随机种子：42
- Batch Size：64
- 优化器：Adam
- 学习率：0.001
- Epoch：10
- 暂时不使用数据增强、Weight Decay 和学习率调度器
- 唯一主要变量仍然是 `width_mult`

# 可复现性检查
- 脚本读取 `split_indices.json` 后会重新计算 SHA-256
- 重新计算结果必须与文件中保存的哈希一致
- 正式划分哈希为 `6242d545f1d70bbd004ba26cd92784461728e0ffb0a64a1f27d1a6421039967e`
- Train DataLoader 使用单独的 `torch.Generator` 控制 shuffle 顺序
- 默认 `num_workers=0`，减少 Windows 多进程加载带来的额外不确定性

# 与 Day28 的关键区别
- Day28 使用全部 50,000 张官方训练图片更新参数
- Day28 每个 Epoch 在官方 Test Set 上计算所谓的 `val_acc`
- Day32 只使用 45,000 张 Train 更新参数，在 5,000 张 Validation 上选择 checkpoint
- Day32 的 CSV 明确保存 `official_test_evaluated=False`
- 因为训练数据减少了 5,000 张，所以 Day28 与 Day32 的准确率不能只归因于数据泄漏，还同时受到训练样本数变化影响

# 代码拆解
- `load_split`：读取索引、检查集合关系并重新计算哈希
- `build_loaders`：构建两个 Train=True 的 CIFAR-10 Dataset，再分别包装为 Train/Validation Subset
- `train_one_epoch`：在 Train Set 上更新参数
- `evaluate`：使用 `model.eval()` 和 `torch.no_grad()` 计算 Validation 指标
- `run_width`：重置随机种子、训练一个宽度并保存最佳权重和历史记录
- `save_summary`：增量保存结果，即使后续宽度训练中断，已完成模型的数据仍然保留

# 运行方式
- 语法检查：`venv\Scripts\python.exe -m py_compile notecode\d32_train_proper_split.py`
- 冒烟测试：`venv\Scripts\python.exe notecode\d32_train_proper_split.py --epochs 1 --widths 0.5 --run-name day32_smoke`
- 正式训练：`venv\Scripts\python.exe notecode\d32_train_proper_split.py --epochs 10`
- 正式训练三个模型可能需要较长时间，不要关闭终端或让电脑进入睡眠

# 输出文件
- 正式结果目录：`checkpoints\day32_proper_width_models\`
- `summary.csv`：三个宽度的参数量、最佳 Val Acc、最佳 Epoch 和训练时间
- `width_x_x\best.pth`：根据 Validation Acc 保存的最佳模型权重
- `width_x_x\history.json`：每个 Epoch 的 Train/Validation Loss 和 Accuracy
- Day32 不应出现 Test Acc；这不是漏记，而是有意保留官方测试集

# 今日自查问题
- 为什么 Day32 只使用 Validation Acc 保存最佳 checkpoint？
- 为什么不能为了获得更好的结果而在训练完成后更换数据划分？
- Train 和 Validation 变换当前相同，为什么还要创建两个 Dataset？
- 为什么 Day28 和 Day32 的准确率不能直接只归因于是否数据泄漏？
- `official_test_evaluated=False` 对最终实验记录有什么意义？

# 实验结果记录
- 冒烟测试使用 `width_mult=0.5` 和 1 Epoch，Train Acc 为 38.02%，Val Acc 为 46.32%，耗时约 60.38 秒
- 冒烟测试确认训练样本为 45,000、验证样本为 5,000，`official_test_evaluated=False`，输出中没有 Test Acc
- 冒烟测试仅用于验证代码链路，不能作为正式模型对照结论
- `width_mult=0.5`：10,875 参数，最佳 Val Acc 为 67.56%，最佳 Epoch 为 10，训练耗时约 531.68 秒
- `width_mult=1.0`：37,579 参数，最佳 Val Acc 为 73.04%，最佳 Epoch 为 7，最终 Val Acc 为 72.64%，训练耗时约 837.33 秒
- `width_mult=1.5`：80,155 参数，最佳 Val Acc 为 74.72%，最佳 Epoch 为 10，训练耗时约 6641.81 秒
- 在规范 Validation 划分上，`1.0` 比 `0.5` 提升 5.48 个百分点，`1.5` 比 `1.0` 仅提升 1.68 个百分点，仍然可以观察到宽度增加后的收益递减
- `width_mult=1.0` 在第 7 轮达到最佳验证结果，之后 Val Acc 回落而 Train Acc 继续上升；第 10 轮训练集与验证集相差约 4.96 个百分点，出现过拟合趋势
- `width_mult=1.5` 第 10 轮训练集与验证集相差约 8.00 个百分点，过拟合倾向比 `1.0` 更明显
- `width_mult=0.5` 第 10 轮训练集与验证集只相差约 0.47 个百分点，但两者都低于宽模型，符合容量不足的表现
- `width_mult=1.5` 的训练耗时约 110.7 分钟，明显高于 `width_mult=1.0` 的约 14.0 分钟；这个差距大于仅根据参数量预期的变化，可能受到 CPU 负载、温度、后台进程或计时环境影响，暂时不能作为稳定的硬件性能结论
- Day32 的 Val Acc 低于 Day28 的 Val Acc 数值不能简单归因于数据泄漏，因为 Day32 同时减少了训练样本，并且使用了不同的数据划分；后续应在统一协议下比较
- 三个结果的 `official_test_evaluated` 均为 `False`，官方测试集仍未用于最终模型选择
- 不要根据 Day28 的 Test Acc 调整 Day32 超参数，否则新的验证协议仍会受到旧测试信息影响
- Day32 结果只负责建立规范 Validation 基线，官方 Test Acc 留到最终实验
