# Day39：综合指标汇总与最终候选模型选择
- Day32 到 Day38 已经分别得到准确率、随机种子稳定性、参数量、FLOPs、真实推理延迟和 SE 消融结果
- 今天不重新训练模型，而是把已有实验结果按照统一协议汇总
- 今天的目标不是寻找某个单项指标最高的模型，而是在准确率、成本、速度和证据可靠性之间做出可解释的选择
- 官方 CIFAR-10 Test Set 仍然不评估，留到最终候选模型确定后使用

# 为什么不能只看准确率
- 单次准确率可能受到随机初始化、数据顺序和训练波动影响
- 参数量影响模型存储和部署空间
- MACs/FLOPs 描述理论计算量，但不能完全代替真实设备速度
- 延迟决定单张图片响应速度，吞吐量更适合批量处理
- 多种子均值和标准差比单次最好结果更能反映稳定性
- 一个科研结论必须同时说明性能收益和额外代价

# Pareto 最优的基本思想
- 本项目希望验证集准确率越高越好，而参数量、FLOPs 和 Batch=1 延迟越低越好
- 如果模型 A 的准确率不低于模型 B，同时参数量、FLOPs 和延迟都不高于模型 B，并且至少一项严格更好，则称 B 被 A 支配
- 没有被其他候选同时在这些指标上压过的模型，构成 Pareto 前沿
- Pareto 前沿不代表唯一最佳模型，它表示不同资源约束下的合理候选
- 选择最终模型还要考虑结果是否来自足够多的随机种子

# 本日汇总的实验来源
- Day32：width=0.5、1.0、1.5 的正式划分训练结果
- Day33/34：width=1.0 的三种子结果和统计量
- Day35：三种宽度的参数量、MACs 和 FLOPs
- Day36：单线程 CPU 上 Batch=1 的中位延迟和 P95
- Day38：width=1.0 的 Baseline 与 Baseline+SE 配对消融
- 所有正式结果使用相同划分 SHA-256：`6242d545f1d70bbd004ba26cd92784461728e0ffb0a64a1f27d1a6421039967e`

# 综合结果
- width=0.5：Best Val Acc `67.56%`，参数 `10,875`，FLOPs `2.879M`，Batch=1 中位延迟 `1.566ms`
- width=1.0：Best Val Acc `73.84% ± 1.30%`，参数 `37,579`，FLOPs `9.896M`，Batch=1 中位延迟 `1.977ms`
- width=1.5：Best Val Acc `74.72%`，参数 `80,155`，FLOPs `21.108M`，Batch=1 中位延迟 `2.521ms`
- width=0.5 和 width=1.5 目前只有 seed=42，属于单种子探索性结果
- width=1.0 是目前唯一完成三种子稳定性验证的宽度

# 速度和复杂度关系
- width=0.5 的 MACs 约为 width=1.0 的 29.1%，但 Batch=1 延迟约为其 79.2%
- width=1.5 的 MACs 约为 width=1.0 的 2.13 倍，但 Batch=1 延迟约增加 27.5%
- 理论 FLOPs 与真实延迟不一定线性对应，内存访问、算子实现和线程调度都会影响结果
- 因此报告中应同时给出理论复杂度和真实设备基准

# 最终候选决策
- 推荐候选：Baseline `width_mult=1.0`
- 原因：准确率明显高于 width=0.5，已经完成三种子验证，同时参数量和延迟远低于 width=1.5
- 严格低延迟备选：Baseline `width_mult=0.5`
- 暂不选择 width=1.5：单次准确率最高，但只有一个 seed，且成本显著增加；需要补做三种子实验后才能声称稳定更优
- 暂不选择 width=1.0+SE：三个 seed 的均值从 `73.84%` 降至 `73.49%`，标准差从 `1.30%` 增至 `1.50%`，训练时间还增加约 `18.27%`
- 这个选择是基于当前证据质量和综合成本，不是声称 width=1.0 在所有硬件或所有数据集上绝对最好

# 代码拆解
- `load_csv`：读取已有实验汇总，并在文件缺失时及时报错
- `parse_day33`：检查三种子、固定划分、width=1.0 和官方测试集标记
- `make_baseline_records`：合并准确率、参数量、FLOPs、延迟和可靠性标签
- `is_dominated`：按照准确率最大化、成本和延迟最小化判断支配关系
- `build_frontier`：生成 Pareto 前沿标记
- `decision.json`：保存最终候选和未入选原因，便于 README 复用

# 运行方式
- 语法检查：`venv\Scripts\python.exe -m py_compile notecode\d39_model_selection.py`
- 正式汇总：`venv\Scripts\python.exe notecode\d39_model_selection.py`
- 脚本只读取 CSV 和 Day38 统计文件，不读取官方 Test Set，也不会重新训练

# 输出文件
- `checkpoints\day39_model_selection\model_summary.csv`：综合指标表
- `checkpoints\day39_model_selection\pareto_frontier.csv`：Pareto 支配关系结果
- `checkpoints\day39_model_selection\accuracy_cost_pareto.png`：参数量与准确率图
- `checkpoints\day39_model_selection\decision.json`：候选选择和排除原因
- `*` 表示单种子探索性结果，不能与三种子结果作完全等价的稳定性比较

# 今日自查
- 为什么 width=1.5 的单次准确率高于 width=1.0，仍然不能直接选它？
- Pareto 前沿中的模型是否只有一个？为什么？
- 为什么理论 FLOPs 更高不一定意味着延迟按相同比例增加？
- 如果目标是移动端实时识别，width=0.5 和 width=1.0 应该如何选择？
- 为什么 Day39 仍然不能使用官方 Test Set 来挑选模型？
- 如何用一句话向导师解释“最终选择 width=1.0 而不是最高准确率的 width=1.5”？
