# Day38：SE 消融实验与配对对照
- Day37 只验证了 SE 模块的张量形状和参数变化，今天训练 Baseline 与 Baseline+SE
- 实验目标：判断 SE 带来的准确率变化是否值得额外参数和计算开销
- 今天不使用 CIFAR-10 官方 Test Set，最终测试集留到最终模型确定后再评估

# 什么是消融实验
- 消融实验通过移除或加入某个组件，观察该组件对整体性能的实际贡献
- Baseline 是不使用 SE 的原始轻量 CNN
- Ablation Model 是在完全相同训练条件下加入 SE 的模型
- 只有控制其他变量不变，准确率差异才有资格归因于 SE

# 为什么采用配对对照
- 每个训练 seed 都对应一个 Baseline 结果和一个 SE 结果
- 对相同 seed 计算 `delta = SE Best Val Acc - Baseline Best Val Acc`
- 配对差值可以减少随机初始化和数据加载顺序差异对比较的影响
- 不能拿 SE 的某一次最好结果与 Baseline 的某一次最差结果直接比较

# 固定实验变量
- 数据划分：Day31 的固定划分，seed=42，Train 45,000，Validation 5,000
- 划分 SHA-256：`6242d545f1d70bbd004ba26cd92784461728e0ffb0a64a1f27d1a6421039967e`
- 训练 seed：42、123、2026
- 模型宽度：`width_mult=1.0`
- Batch Size：64
- 优化器：Adam
- 学习率：0.001
- Epoch：10
- 损失函数：CrossEntropyLoss
- 最优 checkpoint：按 Validation Accuracy 保存
- 唯一结构变量：是否加入 SE，`reduction=16`

# SE 的代价
- Day37 的 width=1.0 Baseline 参数量为 37,579
- Day37 的 width=1.0 SE 模型参数量为 43,431
- SE 额外增加 5,852 个参数，约增加 15.57%
- 参数增加不等于准确率一定提升，还需要结合训练时间、推理延迟和 FLOPs 判断

# 评价指标
- `Best Val Acc`：训练期间最高验证准确率
- `Mean ± Std`：三个 seed 的平均性能和样本标准差
- `delta`：同一个 seed 下 SE 相对于 Baseline 的准确率变化
- `params`：可训练参数量
- `train_seconds`：训练耗时，仅用于成本参考，不能脱离 CPU/GPU 环境比较

# 运行方式
- 语法检查：`venv\Scripts\python.exe -m py_compile notecode\d38_se_ablation.py`
- 冒烟测试：`venv\Scripts\python.exe notecode\d38_se_ablation.py --data-dir data_day38 --epochs 1 --seeds 42 --skip-comparison --run-name checkpoints/day38_smoke`
- 正式实验：`venv\Scripts\python.exe notecode\d38_se_ablation.py --data-dir data_day38 --epochs 10 --seeds 42,123,2026`
- 正式实验依赖 `checkpoints\day33_multiseed\summary.csv`
- 如果只重训 SE，可以使用 `--skip-comparison`，但不能据此得出消融结论

# 输出文件
- `checkpoints\day38_se_ablation\se_summary.csv`：三个 seed 的 SE 结果
- `checkpoints\day38_se_ablation\se\seed_xx\best.pth`：每个 seed 的最佳权重
- `paired_comparison.csv`：按 seed 配对后的 Baseline 与 SE 结果
- `model_statistics.csv`：两种模型的均值、标准差和平均训练时间
- `baseline_se_comparison.png`：均值和样本标准差对比图

# 实际实验结果
- Baseline 三个 seed 的 Best Val Acc：`73.04% / 75.34% / 73.14%`
- SE 三个 seed 的 Best Val Acc：`72.06% / 75.06% / 73.36%`
- 配对 delta：`-0.98 / -0.28 / +0.22` 个百分点
- Baseline：`73.84% ± 1.30%`
- SE：`73.49% ± 1.50%`
- SE 平均准确率变化：`-0.35` 个百分点
- Baseline 平均训练时间：`898.48` 秒
- SE 平均训练时间：`1062.68` 秒，约增加 `18.27%`
- SE 参数量增加：5,852，约 15.57%
- 官方 Test Set：未评估

# 实验结论
- 在本实验的三个 seed 上，SE 平均 Best Val Acc 下降 0.35 个百分点
- 三个配对结果中有两个下降，只有 seed=2026 提升 0.22 个百分点
- SE 的样本标准差为 1.50%，高于 Baseline 的 1.30%，稳定性没有改善
- SE 增加 5,852 个参数，并使平均训练时间增加约 18.27%
- 当前证据不支持把 SE 作为最终模型结构；它仍然是一个规范的负结果消融实验

# 今天自查
- 为什么 Baseline 和 SE 必须使用相同的数据划分？
- 如果 SE 平均准确率提升但标准差变大，应该如何评价？
- 如果准确率只提升 0.3 个百分点但推理延迟增加 30%，是否值得使用？
- 为什么不能把单个 seed 的结果当作 SE 的最终结论？
- 如果 SE 结果下降，如何把这个负结果写成有价值的实验结论？

# Day39 预告
- 汇总 Accuracy、参数量、MACs、Batch=1 延迟和多 seed 稳定性
- 绘制轻量化模型的性能-成本 Pareto 图
- 根据任务目标选择最终模型，Day40 再进行项目总总结和复试材料整理
