# Day40：最终测试评估与项目收尾
- Day39 已经根据验证集、复杂度、真实延迟和稳定性固定最终候选：Baseline `width_mult=1.0`
- 今天才使用 CIFAR-10 官方 Test Set，测试集只用于最终报告，不参与模型结构、超参数或 checkpoint 选择
- 最终评估加载 Day33 三个训练 seed 保存的最佳 checkpoint，不重新训练模型
- 今天完成最终指标、混淆矩阵、README、简历描述和复试问答材料

# 为什么测试集必须最后使用
- 训练集用于更新模型参数
- 验证集用于比较模型、调节超参数和选择最佳 checkpoint
- 测试集用于估计最终模型在未参与决策的数据上的泛化性能
- 如果反复查看测试集结果并据此修改模型，测试集就会逐渐变成验证集，最终准确率会产生乐观偏差
- 本项目在 Day31 到 Day39 始终保持 `official_test_evaluated=False`，Day40 完成模型选择后才打开测试集

# 最终评估流程
- 固定模型结构：`WidthLightVGGSlimGAP(width_mult=1.0)`
- 固定 checkpoint：Day33 的 seed=42、123、2026 最佳验证 checkpoint
- 固定数据：CIFAR-10 `train=False` 官方测试集，共 10,000 张图像
- 固定预处理：ToTensor 后使用训练阶段相同的 CIFAR-10 均值和标准差归一化
- 使用 `model.eval()` 关闭 BatchNorm 的训练行为
- 使用 `torch.inference_mode()` 关闭梯度记录，减少推理开销
- 不执行 `loss.backward()` 和 `optimizer.step()`

# 最终测试结果
- seed=42：Test Accuracy `71.26%`，Test Loss `0.8276`
- seed=123：Test Accuracy `74.10%`，Test Loss `0.7547`
- seed=2026：Test Accuracy `72.71%`，Test Loss `0.8021`
- 三个 seed 均值：`72.69% ± 1.42%`
- 最低/最高：`71.26% / 74.10%`
- 验证集均值：`73.84% ± 1.30%`
- 验证集到测试集的均值变化：`-1.15` 个百分点
- 测试集标准差略高于验证集，说明最终泛化表现存在一定随机种子波动

# 聚合类别结果
- 这里的类别统计聚合了三个 checkpoint，每个类别共 3,000 个样本
- automobile：`86.50%`
- ship：`86.20%`
- truck：`80.50%`
- horse：`77.77%`
- airplane：`75.90%`
- frog：`75.47%`
- deer：`72.77%`
- dog：`59.90%`
- cat：`58.23%`
- bird：`53.67%`
- 最主要的混淆包括 cat -> dog、bird -> deer/cat、dog -> cat、airplane -> ship
- CIFAR-10 的低分辨率和类别间外观相似性，是这些错误的重要原因之一

# 项目最终结论
- 深度可分离卷积将普通卷积拆成 Depthwise Conv 和 Pointwise Conv，显著降低理论计算量
- GAP 分类头避免了大规模全连接层，进一步减少参数量
- width=0.5 速度最快但准确率损失明显
- width=1.5 单次验证准确率较高，但只有一个 seed 且成本显著增加
- width=1.0 在准确率、参数量、FLOPs、延迟和证据可靠性之间取得更均衡的结果
- SE 消融实验没有提升平均准确率，反而增加约 15.57% 参数和约 18.27% 训练时间
- 最终模型不是“单项指标最高”的模型，而是当前实验约束下证据最完整、综合代价合理的模型

# 代码拆解
- `build_loader`：构造官方 Test DataLoader，不进行数据增强
- `evaluate`：累计损失、正确数和 10x10 混淆矩阵
- `class_accuracy`：根据混淆矩阵对每个真实类别计算准确率
- `save_confusion_plot`：把真实标签和预测标签的计数可视化
- `final_report.json`：保存最终测试均值、标准差和类别统计
- `selection_was_completed_before_test=True`：记录测试集确实在模型选择完成后使用

# 运行方式
- 语法检查：`venv\Scripts\python.exe -m py_compile notecode\d40_final_test.py`
- 最终评估：`venv\Scripts\python.exe notecode\d40_final_test.py`
- 当前开发机的备用目录命令：`venv\Scripts\python.exe notecode\d40_final_test.py --data-dir data_day38`
- 脚本默认使用 `checkpoints\day33_multiseed` 下的三个 width=1.0 checkpoint

# 输出文件
- `checkpoints\day40_final_evaluation\test_summary.csv`：逐 seed 测试结果
- `class_accuracy.csv`：聚合类别准确率
- `confusion_matrix.csv`：聚合混淆矩阵数值
- `confusion_matrix.png`：混淆矩阵图
- `final_report.json`：机器可读的最终报告

# 项目交付物
- `README.md`：项目背景、方法、环境、实验协议、结果和结论
- 本地 `docs\interview_qa.md`：复试常见问题和底层原理问答，目录已加入 `.gitignore`
- 本地 `docs\resume_description.md`：简历项目描述和面试自述，目录已加入 `.gitignore`
- `note\d01_note.md` 到 `note\d40_note.md`：按天学习和实验记录

# 最终自查
- 为什么不能根据 Test Accuracy 反过来修改模型？
- 为什么最终报告要同时写均值、标准差和每个 seed？
- 混淆矩阵的行和列分别代表什么？
- 为什么 cat、dog、bird 的准确率低于 automobile、ship？
- 深度可分离卷积减少了哪些乘加计算？它有什么潜在代价？
- 如果导师问“你的创新点是什么”，如何诚实地回答？
- 如果继续研究，下一步应补充哪些实验或方法？
