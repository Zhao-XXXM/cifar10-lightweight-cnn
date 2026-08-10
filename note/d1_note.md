# Tensor 是什么
- Tensor 是 PyTorch 中最核心的数据结构，可以理解为支持 GPU 加速和自动求导的多维数组
- 标量是 0 维 Tensor，向量是 1 维 Tensor，矩阵是 2 维 Tensor，图像 Batch 通常是 4 维 Tensor
- CIFAR-10 图像送入 CNN 后，常见形状是 `(Batch, Channel, Height, Width)`

# requires_grad 的作用
- `requires_grad=True` 表示该 Tensor 需要被 PyTorch 记录进计算图
- 不加 `requires_grad` 时，Tensor 只参与普通数值计算，不会记录梯度信息
- 代码中 `y1.grad_fn` 为 `None`，而 `y2.grad_fn` 会显示乘法对应的反向传播函数

# 计算图与反向传播
- 计算图记录了 Tensor 之间的运算依赖关系
- 反向传播本质：从最终结果出发，沿计算图反向遍历，用链式法则逐层计算梯度
- `z.backward()` 的含义是：从 z 这个终点开始，自动计算所有叶子节点对 z 的偏导数

# 梯度只保存在叶子节点
- 叶子节点通常是用户直接创建并设置 `requires_grad=True` 的 Tensor
- 梯度会保存在叶子节点的 `.grad` 属性中
- 中间变量默认不保留 `.grad`，因为训练时真正需要更新的是模型参数

# 手算梯度例子
- 若 `z = 3x + 1`，则 `dz/dx = 3`
- 若 `c = a * b + a ** 2`，则 `dc/da = b + 2a`，`dc/db = a`
- 代码中的自动求导结果可以和手算结果互相验证

# 梯度累加
- PyTorch 中梯度默认累加，而不是每次自动覆盖
- 这是为了支持梯度累积、多任务损失叠加等工程技巧
- 在训练循环中必须手动执行 `zero_grad()`，否则梯度会越积越大，导致训练不稳定

# 环境记录
- PyTorch 版本可通过 `torch.__version__` 查看
- 当前项目后续运行建议使用项目虚拟环境中的 Python

# 今日自查问题
- 为什么只有设置 `requires_grad=True` 的 Tensor 才会进入计算图？
- `backward()` 是从计算图的起点开始，还是从终点开始？
- 为什么训练循环中每轮都要清空梯度？
