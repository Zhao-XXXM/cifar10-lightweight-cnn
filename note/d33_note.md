# Day33：多随机种子稳定性实验
- Day32 在固定 `seed=42` 下比较了三个宽度模型
- 单次训练结果可能受到随机初始化、DataLoader shuffle 和硬件非确定性的影响
- Day33 固定模型结构为 `width_mult=1.0`，只改变训练随机种子 `42、123、2026`
- 选择 `width_mult=1.0` 的原因是它在 Day32 中具有更均衡的精度与资源成本，且 `width_mult=1.5` 的训练计时出现了需要复测的异常
- 今天保存每个种子的原始结果，均值和标准差放到 Day34 计算

# 随机性来自哪里
- 模型参数初始化通常由伪随机数生成器决定
- Train DataLoader 的 shuffle 顺序会改变每个 Batch 的组成
- 不同 Batch 顺序会改变梯度估计和参数更新轨迹
- GPU 某些并行算子可能存在非确定性，CPU 也可能受到线程调度影响
- 因此同一结构、同一数据和同一超参数也可能得到略有差异的结果

# 为什么必须固定数据划分
- Day33 的研究问题是“训练随机性会造成多大波动”
- 如果每个种子同时使用不同的 Train/Validation 划分，就无法区分波动来自训练过程还是样本组成
- 所有种子统一读取 Day31 的 `seed=42/split_indices.json`
- 训练种子只控制模型初始化和训练过程，不改变 Validation 样本

# 多次实验的统计量
- 对多个种子的准确率 `a_1, a_2, ..., a_n`，均值为：`mean = 1/n * sum(a_i)`
- 样本标准差可以描述不同种子之间的波动：`std = sqrt(1/(n-1) * sum((a_i - mean)^2))`
- 均值越高通常表示平均性能越好，标准差越小通常表示结果更稳定
- 三个种子只能提供初步稳定性证据，不能当作严格的统计显著性检验
- Day34 会统一计算 `mean ± std`，Day33 只负责产生可信的原始记录

# 公平实验设置
- 模型：`WidthLightVGGSlimGAP(width_mult=1.0)`
- 数据划分：Day31 `seed=42`，Train 45,000，Validation 5,000
- 训练种子：42、123、2026
- Batch Size：64
- 优化器：Adam
- 学习率：0.001
- Epoch：10
- 不使用官方 Test Set

# 代码拆解
- `parse_seeds`：解析并检查随机种子列表不能重复
- `load_split`：读取并校验固定的数据划分和 SHA-256
- `SimpleNamespace`：为每个训练种子构造与 Day32 相同的训练参数对象
- `run_width`：复用 Day32 的训练、验证、checkpoint 和 history 逻辑
- `seed_xx/summary.csv`：保存单个种子的结果
- `day33_multiseed/summary.csv`：汇总所有种子的原始结果

# 运行方式
- 语法检查：`venv\Scripts\python.exe -m py_compile notecode\d33_multiseed.py`
- 冒烟测试：`venv\Scripts\python.exe notecode\d33_multiseed.py --epochs 1 --seeds 42 --run-name day33_smoke`
- 正式实验：`venv\Scripts\python.exe notecode\d33_multiseed.py --epochs 10 --seeds 42,123,2026`
- 若需要单独复现实验：`venv\Scripts\python.exe notecode\d33_multiseed.py --epochs 10 --seeds 123`

# 输出文件
- 正式目录：`checkpoints\day33_multiseed\`
- `summary.csv`：三个训练种子的原始 Validation 结果
- `seed_xx\best.pth`：该训练种子的最佳 Validation checkpoint
- `seed_xx\history.json`：逐 Epoch 训练与验证记录
- `official_test_evaluated` 应保持为 `False`

# 今日自查问题
- 为什么多种子实验必须固定 Train/Validation 划分？
- 为什么只运行一次不能充分证明模型优于另一个模型？
- 标准差较大说明了什么？应该优先怀疑模型、数据量还是训练过程？
- 为什么 Day33 不同时比较多个宽度和多个种子？
- 如果某个种子结果特别高或特别低，应该删除它还是保留并分析？

# 实验结果记录
- 冒烟测试使用 `seed=42` 和 1 Epoch，Train Acc 为 45.92%，Val Acc 为 54.98%，与 Day32 同配置第 1 轮结果一致
- 冒烟测试确认按种子保存 checkpoint、history 和 summary，且 `official_test_evaluated=False`
- 冒烟测试只验证复现链路，不参与正式多种子统计
- `seed=42`：最佳 Val Acc 为 73.04%，最佳 Epoch 为 7，最终 Val Acc 为 72.64%，训练耗时约 826.22 秒
- `seed=123`：最佳 Val Acc 为 75.34%，最佳 Epoch 为 10，最终 Val Acc 为 75.34%，训练耗时约 837.16 秒
- `seed=2026`：最佳 Val Acc 为 73.14%，最佳 Epoch 为 9，最终 Val Acc 为 72.72%，训练耗时约 1032.06 秒
- 三个结果的 `split_sha256` 完全一致，`official_test_evaluated` 全部为 `False`
- 三个种子的最佳 Val Acc 最高与最低相差 2.30 个百分点，说明单次训练结果存在不可忽略的随机波动
- 不在 Day33 直接下最终结论，Day34 统一计算均值、标准差并绘图
