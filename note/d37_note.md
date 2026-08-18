# Day37：SE 通道注意力模块
- Day35 和 Day36 分别量化了理论计算量和真实推理性能
- Day37 设计一个与轻量 CNN 兼容的注意力改进：Squeeze-and-Excitation（SE）
- 今天只验证模块数学逻辑、张量形状和参数增加比例，不训练模型
- Day38 才进行 Baseline 与 Baseline+SE 的公平训练对照

# 为什么需要通道注意力
- 卷积层输出包含多个通道，每个通道可以看作一种特征响应
- 不同图片和不同阶段中，各通道的重要程度可能不同
- 普通卷积会生成通道特征，但不会显式地为每个样本自适应调整通道权重
- SE 通过全局空间信息学习每个通道的权重，再重新缩放特征图

# Squeeze：空间压缩
- 输入特征图记为 `X`，形状为 `(B, C, H, W)`
- 对每个通道做全局平均池化：`z_c = 1/(H*W) * sum_{h,w} X[c,h,w]`
- 得到 `z`，形状为 `(B, C)`
- 这一步把空间信息压缩成每个通道的全局响应描述
- 这里的 Squeeze 与分类头中的 GAP 都使用空间平均，但用途不同：SE 的 GAP 生成通道权重，分类头 GAP 直接服务于分类

# Excitation：学习通道权重
- 先用降维 Linear：`C -> C/r`
- 经过 ReLU 引入非线性
- 再用升维 Linear：`C/r -> C`
- 最后用 Sigmoid 把每个通道权重压到 `(0, 1)`
- 公式为：`s = sigmoid(W2 * ReLU(W1 * z))`
- `r` 是 reduction ratio，用于控制 SE 的额外参数量

# Scale：重新加权
- 将 `s` reshape 为 `(B, C, 1, 1)`
- 对输入特征逐通道相乘：`Y[c,h,w] = s_c * X[c,h,w]`
- 权重接近 1 表示保留该通道，权重接近 0 表示抑制该通道
- SE 不改变特征图的空间尺寸、批大小和通道数

# SE 的额外参数量
- 忽略 bias 时，两层 Linear 的参数量约为：`C * (C/r) + (C/r) * C = 2C^2/r`
- 实际代码使用 Linear bias，因此还会增加两个 bias 向量
- `r` 越大，SE 参数越少，但通道权重生成网络的表达能力也更弱
- 当前 `reduction=16`，优先控制轻量化成本
- SE 的参数量与通道数平方有关，因此宽模型中的 SE 开销会更明显

# SE 的插入位置
- 当前每个 `DepthwiseSeparableConv` 后接一个 SEBlock
- 执行顺序为：Depthwise Conv -> Pointwise Conv -> BatchNorm/ReLU -> SE -> 下一层
- SE 位于非线性特征提取之后，可以根据当前样本重新调整通道响应
- 每个 VGG block 包含两次 Depthwise Separable Conv，因此每个 block 有两个 SEBlock

# 为什么这是轻量化路线中的合理改进
- SE 不增加特征图空间尺寸，也不增加普通 3 * 3 卷积
- 额外计算主要是全局池化、两个小型 Linear 和逐元素乘法
- 它可能改善通道选择能力，但会增加参数、MACs 和延迟
- Day38 必须同时报告准确率提升和额外成本，不能只看 Val Acc

# 代码拆解
- `SEBlock`：实现 Squeeze、Excitation 和 Scale 三步
- `hidden_channels=max(1, channels // reduction)`：避免小宽度下隐藏通道变成 0
- `view(x.size(0), channels, 1, 1)`：把通道权重变成可广播的形状
- `SEWidthLightVGGBlock`：在两个 Depthwise Separable Conv 后分别插入 SE
- `count_se_params`：只统计 SE 模块新增的参数量
- `inspect_width`：用 `(2,3,32,32)` dummy input 检查输出形状和数值有限性

# 运行方式
- 语法检查：`venv\Scripts\python.exe -m py_compile notecode\d37_se_attention.py`
- 正式结构验证：`venv\Scripts\python.exe notecode\d37_se_attention.py`
- 只验证均衡宽度：`venv\Scripts\python.exe notecode\d37_se_attention.py --widths 1.0`
- 修改 reduction：`venv\Scripts\python.exe notecode\d37_se_attention.py --reduction 8`
- Day37 不读取 CIFAR-10，也不进行训练

# 输出文件
- 输出目录：`checkpoints\day37_se_structure\`
- `se_structure.csv`：Baseline、SE 新增参数和总参数
- `se_structure.json`：结构化记录

# 今日自查问题
- 为什么 Squeeze 使用全局平均池化后仍然能保留通道重要性信息？
- reduction ratio 为什么能降低 SE 的参数量？
- 为什么 SE 不改变特征图的空间尺寸？
- 如果 SE 提升了准确率但增加了 20% 延迟，是否一定值得使用？
- 为什么 Day38 必须让 Baseline 和 SE 使用完全相同的数据划分、种子和训练策略？

# 实验结果记录
- `width_mult=0.5`：Baseline 为 10,875 参数，SE 新增 1,582 参数，总参数量为 12,457，增加约 14.55%
- `width_mult=1.0`：Baseline 为 37,579 参数，SE 新增 5,852 参数，总参数量为 43,431，增加约 15.57%
- `width_mult=1.5`：Baseline 为 80,155 参数，SE 新增 12,810 参数，总参数量为 92,965，增加约 15.98%
- 三个宽度的 SE 模型输出形状均为 `(2, 10)`，且没有 NaN/Inf
- SE 参数增加比例随宽度略有上升，符合 SE 两层 Linear 参数与通道数平方相关的规律
- 不把结构验证结果当成准确率提升，准确率结论留到 Day38
