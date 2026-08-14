# Day30：单张图片推理与 Top-K 预测
- Day29 统计了整个测试集的类别准确率和混淆关系
- Day30 把分析对象缩小到一张具体图片，观察模型对这个样本的完整判断过程
- 今天关注三个问题：模型的 Top-1 预测是什么、Top-3 候选是什么、不同宽度模型对同一张图片是否一致
- 单张推理是连接“训练指标”和“实际使用过程”的基础，也是后续错误案例分析的入口

# 分类模型输出的数学含义
- 模型最后一层输出长度为 10 的 logits 向量：`z = [z_0, z_1, ..., z_9]`
- logits 是未归一化的类别分数，可以是任意实数，不直接等于概率
- Softmax 将 logits 转为概率：`p_i = exp(z_i) / sum_j exp(z_j)`
- 所有类别概率之和为 1，概率越大表示模型相对更偏向该类别
- Top-1 是概率最大的类别，也就是 `argmax(p)` 的结果
- Top-3 是概率最大的三个类别及其排序，适合观察模型的第二候选和第三候选

# 为什么不能只看 Top-1
- Top-1 只能回答“模型最终选了谁”，不能说明第二候选是谁
- 如果真实类别没有出现在 Top-1，但出现在 Top-3，说明模型已经提取到部分相关特征，只是类别边界还不够稳定
- 如果真实类别完全不在 Top-3，说明当前样本对模型来说更困难，应该结合图片内容和错误类型继续分析
- Top-K 不是替代准确率的指标，它是对单样本预测过程的补充观察

# 推理时必须保持预处理一致
- 训练时使用了 `ToTensor()` 和 CIFAR-10 均值、标准差归一化
- 推理时必须使用相同的归一化，否则输入分布会发生变化
- 模型输入形状仍然是 `(1, 3, 32, 32)`，其中 `1` 表示一次只推理一张图片
- 代码为可视化单独保留未归一化图片；如果只有模型输入张量，就需要先反归一化再显示，否则颜色会异常
- 自定义图片会被转换为 RGB 并缩放到 `32 * 32`，但自然图片与 CIFAR-10 的数据分布可能不同，结果只能作为演示

# 不同宽度模型的公平比较
- 三个模型使用同一张图片、同一套预处理和各自 Day28 的最佳 checkpoint
- 如果三个模型 Top-1 一致，说明该样本的主要特征比较稳定
- 如果 Top-1 不一致，可以比较它们的 Top-3 列表和概率差距
- 不能把单张图片上的预测差异直接解释为总体性能差异，单样本不具有统计代表性

# 代码拆解
- `build_transforms`：分别构建模型输入变换和用于显示的变换
- `load_sample`：从 CIFAR-10 测试集索引或自定义图片中读取样本
- `predict`：执行前向传播、Softmax 和 `torch.topk`
- `save_results_csv`：保存每个宽度模型的 Top-K 预测结果
- `save_visualization`：左侧显示原图，右侧显示不同模型的概率条形图
- `summary.json`：保存样本真实标签和各模型预测结果

# 运行方式
- 语法检查：`venv\Scripts\python.exe -m py_compile notecode\d30_single_image_inference.py`
- 默认使用测试集第 0 张图片：`venv\Scripts\python.exe notecode\d30_single_image_inference.py`
- 更换测试集样本：`venv\Scripts\python.exe notecode\d30_single_image_inference.py --index 1234`
- 只分析均衡模型：`venv\Scripts\python.exe notecode\d30_single_image_inference.py --widths 1.0 --index 1234`
- 调整 Top-K：`venv\Scripts\python.exe notecode\d30_single_image_inference.py --index 1234 --topk 5`
- 分析自己的图片：`venv\Scripts\python.exe notecode\d30_single_image_inference.py --image path\to\image.jpg`

# 输出文件
- 输出目录：`checkpoints\day30_single_inference\`
- `index_x_predictions.csv`：各模型的 Top-K 类别和概率
- `index_x_predictions.png`：原图与概率条形图
- `index_x_summary.json`：结构化预测摘要
- 使用 `--image` 时，文件名中的 `index_x` 会改为 `custom`

# 实验结果记录
- `index=0` 的真实类别为 `cat`，三个模型的 Top-1 都是 `cat`；Top-3 中都包含 `dog`，说明模型虽然判断正确，但第二候选仍然是语义相近的动物类别
- `index=1` 的真实类别为 `ship`，三个模型的 Top-1 都是 `ship`；`width_mult=1.5` 的概率最高，为 99.74%，但这只说明它在这一张图片上更自信，不能直接推出整体校准更好
- `index=10` 的真实类别为 `airplane`，三个模型的 Top-1 都是 `airplane`；Top-3 候选中出现了 `deer`、`bird`、`dog` 等不同类别，说明不同宽度模型对次级候选的排序并不完全一致
- `index=100` 的真实类别为 `deer`：`width_mult=0.5` 预测正确且概率为 59.56%，`width_mult=1.0` 错误预测为 `horse` 且概率为 73.49%，`width_mult=1.5` 预测正确且概率为 88.23%
- 在 `index=100` 中，`width_mult=1.0` 的真实类别 `deer` 排在 Top-2，说明模型提取到了部分相关特征，但最终被视觉上相近的 `horse` 超过
- 选取的 4 张图片中，三个模型共有 12 次 Top-1 预测，其中 11 次正确；这个样本数量太少，不能把它当成模型准确率，只能用于观察具体推理行为
- `index=100` 的错误与 Day29 中动物类别的混淆现象一致，说明单样本可视化能够帮助解释混淆矩阵中的统计结果
- 本次结果说明：总体准确率更高的模型不一定在每一张图片上都更好；单样本推理适合发现模型分歧和形成改进假设，不适合替代完整测试集评估
- 不要把 Softmax 概率直接称为模型“绝对可信度”，它只能表示当前类别分数的相对归一化结果

# 今天的自查问题
- 为什么 logits 不能直接当成概率？
- 为什么推理时必须调用 `model.eval()`？它会影响哪些层的行为？
- 为什么显示图片要使用未归一化张量，而送入模型必须使用与训练阶段一致的归一化张量？
- 如果真实标签排在 Top-3 的第二位，应该怎样描述这个预测结果？
- 为什么不能根据一张图片的预测结果判断哪个宽度模型整体更好？