# 三模型对照实验
- Day26 将 VGG-Slim、LightVGG-Slim 和 LightVGG-Slim-GAP 放在同一张实验表中比较
- 对照实验的目标不是寻找单一最高准确率，而是观察准确率、参数量和模型大小之间的权衡
- 当前三种模型都使用 CIFAR-10 和 10 Epoch 训练结果

# 三种模型
- VGG-Slim：普通 3 * 3 卷积 + Flatten 分类头
- LightVGG-Slim：Depthwise Separable Convolution + Flatten 分类头
- LightVGG-Slim-GAP：Depthwise Separable Convolution + Global Average Pooling 分类头

# 对比指标
- 参数量：模型中可训练参数的总数
- checkpoint 大小：保存的最佳权重文件大小
- 最佳 Val Acc：训练过程中验证集最高准确率
- 最佳 Epoch：达到最高验证准确率的训练轮次
- 训练时间：完整训练所需时间；如果历史实验没有保存该字段，就不能凭空补写

# 当前可用数据
- VGG-Slim：已有 `vgg_slim_best.pth`，但缺少 `vgg_slim_history.json`
- LightVGG-Slim：已有完整 10 Epoch 历史记录，最高 Val Acc 为 72.18%
- LightVGG-Slim-GAP：已有完整 10 Epoch 历史记录，最高 Val Acc 为 74.88%
- 因此 VGG-Slim 的准确率需要通过重新评估已保存 checkpoint 得到，最佳 Epoch 和训练耗时不能凭空补写

# 当前对照结果
- VGG-Slim 已保存 checkpoint 重新评估 Val Acc 为 83.15%
- LightVGG-Slim 最高 Val Acc 为 72.18%
- LightVGG-Slim-GAP 最高 Val Acc 为 74.88%
- 这说明当前轻量模型没有超过 VGG-Slim 的绝对准确率，但显著降低了参数量和 checkpoint 体积

# 参数压缩比例
- 相对 VGG-Slim 的参数压缩比例可以计算为：`1 - 新模型参数量 / VGG-Slim 参数量`
- LightVGG-Slim 相对 VGG-Slim 约减少 30.87% 参数
- LightVGG-Slim-GAP 相对 VGG-Slim 约减少 95.39% 参数
- 参数减少不等于准确率必然下降，最终还要结合训练稳定性和多次实验判断

# 准确率/参数量指标
- 可以用 `Val Acc / 参数量(M)` 粗略衡量参数利用效率
- 这个指标不是严格的工程性能指标，因为它没有考虑 FLOPs、内存访问和实际推理速度
- 它适合做初步结构比较，不能替代真实设备上的延迟测试

# 当前阶段实验结论
- Depthwise Separable Convolution 显著压缩了特征提取部分的参数量
- GAP 进一步消除了大 Flatten 分类头带来的参数开销
- 当前结果显示，VGG-Slim 准确率最高，LightVGG-Slim-GAP 参数效率最高
- 因此这个阶段的重点不是“轻量模型击败大模型”，而是理解轻量化带来的体积优势与精度代价
- 仍然需要注意：VGG-Slim 的训练历史没有保存，后续应统一重新测量或明确标记为未记录

# 科研表达方式
- 不要只说“我的模型参数少、准确率高”
- 更规范的表述是：“在当前实验中，VGG-Slim checkpoint 重新评估准确率为 83.15%，而 LightVGG-Slim-GAP 将参数量压缩到 37,579 个并取得 74.88% 的最高验证准确率，体现了轻量化模型在参数效率和模型体积上的优势。”

# 今日自查问题
- 为什么三种模型必须放在同一张表里比较？
- 参数量减少 95% 是否意味着推理速度也提高 95%？
- 为什么没有记录过的训练时间不能直接估算？
- 当前结果能否直接证明 GAP 一定优于 Flatten？
