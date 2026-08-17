# Day35：参数量、MACs 与 FLOPs
- Day28-Day34 主要分析准确率、错误类型和随机种子稳定性
- Day35 进入模型效率分析，回答“模型每次处理一张图片需要多少计算量”
- 今天统计三个宽度模型的可训练参数量、FP32 理论存储大小、MACs 和 FLOPs
- 复杂度统计只依赖模型结构与输入尺寸，不依赖训练数据和 checkpoint

# 参数量与计算量不是同一个指标
- 参数量表示模型需要存储和学习多少权重
- MACs 表示一次前向推理中大约执行多少次乘加运算
- FLOPs 表示浮点运算次数，本文约定一次乘法和一次加法计为 2 FLOPs
- 参数量影响模型存储、加载和部分内存访问成本
- MACs/FLOPs 更直接描述理论计算量，但不等于真实运行时间

# 普通卷积的参数量
- 普通卷积参数量为：`Cin * Cout * Kh * Kw`
- 如果有 bias，还需要加上 `Cout`
- 本项目的 Conv2d 使用 `bias=False`，因为后面接了 BatchNorm

# 分组卷积的参数量
- 分组卷积把输入和输出通道划分成 `groups` 组
- 参数量为：`Cout * (Cin / groups) * Kh * Kw`
- Depthwise 卷积是特殊情况：`groups=Cin` 且通常 `Cout=Cin`
- Depthwise 3 * 3 卷积参数量约为：`Cin * 3 * 3`
- Pointwise 1 * 1 卷积参数量为：`Cin * Cout`

# Depthwise Separable Conv 的计算量
- 普通卷积 MACs 约为：`Hout * Wout * Cout * Cin * Kh * Kw`
- 分组卷积 MACs 约为：`Hout * Wout * Cout * (Cin / groups) * Kh * Kw`
- 本项目的 Depthwise 和 Pointwise 是两个独立 Conv2d，脚本会分别统计后相加
- 不能把整个 Depthwise Separable Conv 错误地按一个普通卷积计算

# Linear 和池化层
- Linear 每个样本的 MACs 为：`in_features * out_features`
- `AdaptiveAvgPool2d`、ReLU、BatchNorm 和 MaxPool 的精确算力统计需要额外约定
- 本次主表只统计 Conv2d 与 Linear 的 MACs，保证定义清晰且便于横向比较
- 因此这里的 FLOPs 是“卷积与全连接部分的理论 FLOPs”，不是硬件 profiler 的完整指令数

# 为什么使用 Hooks
- 前向 Hook 可以在模块实际执行时读取输入和输出形状
- 输出空间尺寸可能受到 stride、padding 和池化影响，不能只看静态输入尺寸猜测
- 脚本遍历 `named_modules()`，只给 Conv2d 和 Linear 注册 Hook
- Hook 记录层名称、类型、groups、输入形状、输出形状、MACs 和 FLOPs
- 统计结束后必须移除 Hook，避免重复运行时累计回调

# 复杂度统计的公平条件
- 三个模型都使用输入形状 `(1, 3, 32, 32)`
- 三个模型的网络深度、卷积核和池化结构相同
- 唯一变化是 `width_mult`
- 因此参数量和 MACs 的差异可以主要归因于通道宽度

# 复杂度与宽度的关系
- 主要 Pointwise Conv 同时受输入和输出通道影响，计算量通常接近 `alpha^2` 缩放
- 第一层输入通道固定为 3，最后 Linear 输出类别固定为 10
- 因此实际总 MACs 和参数量不会严格等于 `alpha^2` 倍
- 宽度增大通常会提高表达能力，也会增加存储和计算成本

# 代码拆解
- `count_layers`：注册 Hook 并执行一次 dummy forward
- Conv2d 公式：使用输出空间、输出通道、分组后的输入通道和卷积核大小计算 MACs
- Linear 公式：使用 `in_features * out_features` 计算 MACs
- `layer_details.csv`：保存每一层的详细复杂度，便于定位主要计算开销
- `complexity.csv`：保存每个宽度模型的总参数量、MACs 和 FLOPs
- `complexity_comparison.png`：绘制参数量和 MACs 的横向比较

# 运行方式
- 语法检查：`venv\Scripts\python.exe -m py_compile notecode\d35_complexity_profile.py`
- 正式统计：`venv\Scripts\python.exe notecode\d35_complexity_profile.py`
- 只统计一个模型：`venv\Scripts\python.exe notecode\d35_complexity_profile.py --widths 1.0`
- Day35 不需要读取 CIFAR-10 数据集，也不需要训练模型

# 输出文件
- 输出目录：`checkpoints\day35_complexity_profile\`
- `complexity.csv`：模型级复杂度汇总
- `layer_details.csv`：逐层复杂度明细
- `complexity.json`：结构化汇总结果
- `complexity_comparison.png`：参数量和 MACs 对比图

# 统计口径注意事项
- 本项目将 1 MAC 约定为 1 次乘法加 1 次加法，并将其换算为 2 FLOPs
- 不同工具可能把 MACs 和 FLOPs 采用不同定义，写报告时必须说明统计口径
- 理论 FLOPs 少不保证真实延迟低，真实速度还受内存访问、线程、缓存和算子实现影响
- Day36 会在实际 CPU 上测量 Batch=1 和 Batch=64 的延迟与吞吐量

# 今日自查问题
- 为什么 Depthwise 卷积不能按普通卷积公式计算？
- 为什么 `width_mult=0.5` 的参数量不会严格等于 `width_mult=1.0` 的四分之一？
- 为什么 MACs/FLOPs 少不一定代表真实推理时间短？
- 为什么本次统计不把 BatchNorm、ReLU 和池化算入主 MACs？
- 如果两个模型参数量接近，但 MACs 差异很大，应该如何解释？

# 实验结果记录
- `width_mult=0.5`：10,875 参数，FP32 理论大小约 0.041 MB，1.439M MACs，2.879M FLOPs
- `width_mult=1.0`：37,579 参数，FP32 理论大小约 0.143 MB，4.948M MACs，9.896M FLOPs
- `width_mult=1.5`：80,155 参数，FP32 理论大小约 0.306 MB，10.554M MACs，21.108M FLOPs
- 相比 `width_mult=1.0`，`0.5` 的 MACs 减少约 70.91%，`1.5` 的 MACs 增加约 113.29%，与参数量变化趋势基本一致
- `width_mult=1.0` 中 MACs 最大的三层都是 Pointwise `1 * 1` 卷积，每层约 1.049M MACs，说明主要计算开销来自通道混合，而不是 Depthwise 空间卷积
- `layer_details.csv` 中的 Depthwise 层使用 `groups=Cin`，统计公式正确地除以分组数，没有按普通卷积高估计算量
- 三个宽度的 MACs 与参数量都没有严格按 `alpha^2` 变化，但比例非常接近，原因是第一层输入通道和最后分类输出固定
- 当前结果只代表理论 Conv2d/Linear MACs，不代表真实推理延迟；真实延迟留到 Day36 测量
- 不把理论 FLOPs 直接当成真实延迟，Day36 使用独立基准测试验证
