# Day36：真实推理延迟与吞吐量
- Day35 计算了参数量、MACs 和理论 FLOPs
- Day36 在实际设备上运行模型前向传播，测量延迟和吞吐量
- Batch=1 主要反映单张图片的响应时间
- Batch=64 主要反映批量处理时的吞吐能力
- 理论复杂度与真实速度需要同时报告，不能相互替代

# 延迟与吞吐量
- 延迟表示完成一次前向推理所需时间，单位通常为毫秒
- Batch 延迟是整个 Batch 完成推理的时间
- 单图平均延迟为：`batch_latency / batch_size`
- 吞吐量为：`batch_size / batch_time`，本项目使用 images/s
- 低延迟适合实时或交互任务，高吞吐量适合离线批处理任务

# 为什么要预热
- 第一次前向传播可能包含内存分配、算子初始化和缓存建立
- 如果直接记录第一次时间，结果通常不能代表稳定运行状态
- 预热阶段执行若干次前向传播，但不加入正式统计
- 本项目默认预热 10 次，再正式测量 50 次

# 为什么使用中位数和 P95
- 平均值容易被后台任务、系统调度等偶发慢请求影响
- 中位数表示 50% 的测量不超过该时间，适合描述典型延迟
- P95 表示 95% 的测量不超过该时间，适合观察尾部延迟
- P95 明显高于中位数时，说明运行时间存在较大抖动
- 只有重复测量才能同时观察典型值和尾部情况

# CPU 线程数
- CPU 推理速度受线程数量影响，线程越多并不保证线性加速
- 不记录线程数就无法公平复现实验
- Day36 默认使用 `torch.set_num_threads(1)`，得到单线程基准
- 后续可以额外测试 2 或 4 线程，但必须单独记录，不能混在同一张表中

# CUDA 计时为什么需要同步
- CUDA 算子通常异步提交，Python 调用返回时 GPU 可能尚未完成计算
- 只测 Python 调用时间会严重低估真实 GPU 延迟
- 在 CUDA 模式下，脚本会在计时前后执行 `torch.cuda.synchronize()`
- CPU 模式不需要这一步

# 本次计时包含与不包含的内容
- 包含：模型 Conv2d、BatchNorm、ReLU、Pooling、GAP 和 Linear 的前向传播
- 不包含：磁盘读取、PIL/OpenCV 解码、Resize、Normalize 和数据传输
- 输入张量在计时前已经创建并放到目标设备
- 因此结果是“纯模型前向延迟”，不是完整应用端到端延迟

# 为什么加载训练权重
- 结构相同的随机权重与训练权重理论计算量一致
- 脚本仍加载 Day32 的最佳 checkpoint，确保基准对象与实验模型完全对应
- 权重加载在计时前完成，不计入推理延迟
- 官方 Test Set 不会被读取或评估

# 公平基准测试条件
- 三个模型输入尺寸均为 `(Batch, 3, 32, 32)`
- 使用相同设备、线程数、预热次数和重复次数
- 使用 `model.eval()` 关闭训练模式
- 使用 `torch.inference_mode()` 关闭梯度记录
- 模型和输入张量都在计时开始前放到目标设备

# 代码拆解
- `resolve_device`：选择 CPU 或 CUDA，并检查设备是否可用
- `synchronize`：仅在 CUDA 下等待异步计算完成
- `benchmark`：执行预热和重复前向计时
- `percentile`：排序后取向上取整位置，计算经验 P95
- `benchmark_summary.csv`：保存中位延迟、P95、单图延迟和吞吐量
- `latency_samples.csv`：保存每一次原始计时，便于检查异常值
- `environment.json`：记录系统、PyTorch、设备、线程和统计口径

# 运行方式
- 语法检查：`venv\Scripts\python.exe -m py_compile notecode\d36_inference_benchmark.py`
- 冒烟测试：`venv\Scripts\python.exe notecode\d36_inference_benchmark.py --widths 1.0 --batch-sizes 1 --warmup 2 --repeats 5 --output day36_smoke`
- 正式 CPU 基准：`venv\Scripts\python.exe notecode\d36_inference_benchmark.py`
- 额外测试 4 线程：`venv\Scripts\python.exe notecode\d36_inference_benchmark.py --num-threads 4 --output day36_threads4`
- 如果有 CUDA：`venv\Scripts\python.exe notecode\d36_inference_benchmark.py --device cuda --output day36_cuda`

# 输出文件
- 正式目录：`checkpoints\day36_inference_benchmark\`
- `benchmark_summary.csv`：模型级延迟和吞吐量汇总
- `latency_samples.csv`：每次重复的原始延迟
- `environment.json`：软硬件和基准设置
- `benchmark_comparison.png`：单图延迟与吞吐量对比图

# 结果分析原则
- 先比较 Batch=1 的中位延迟，再观察 P95 是否稳定
- 比较 Batch=64 的吞吐量，不要只看整批耗时
- 检查真实速度排序是否与 Day35 MACs 排序一致
- 如果排序不一致，可能与内存访问、算子优化、线程调度或测量噪声有关
- 不使用 Day32 的训练时间代替推理延迟，两者包含的计算过程完全不同

# 今日自查问题
- 为什么第一次前向传播不适合直接计入统计？
- Batch=1 延迟和 Batch=64 吞吐量分别适合描述什么场景？
- 为什么 CUDA 计时必须同步？
- 为什么理论 FLOPs 更少的模型不一定在实际设备上更快？
- 为什么训练时间不能代替推理延迟？

# 实验结果记录
- 测试环境为 Windows 11、PyTorch 2.13.0+cpu、AMD64 CPU，使用 1 个 PyTorch 计算线程、10 次预热和 50 次正式重复
- `width_mult=0.5`：Batch=1 中位延迟 1.566 ms，P95 为 1.729 ms；Batch=64 中位整批延迟 35.399 ms，单图约 0.553 ms，吞吐量约 1807.97 images/s
- `width_mult=1.0`：Batch=1 中位延迟 1.977 ms，P95 为 2.227 ms；Batch=64 中位整批延迟 62.286 ms，单图约 0.973 ms，吞吐量约 1027.52 images/s
- `width_mult=1.5`：Batch=1 中位延迟 2.521 ms，P95 为 2.857 ms；Batch=64 中位整批延迟 108.283 ms，单图约 1.692 ms，吞吐量约 591.04 images/s
- 真实速度排序与 Day35 MACs 排序一致：`0.5` 最快，`1.0` 居中，`1.5` 最慢
- `width_mult=1.5` 的 MACs 是 `1.0` 的约 2.13 倍，但 Batch=1 中位延迟只增加约 27.5%，说明理论计算量与真实延迟不是线性关系
- 相比 Batch=1，Batch=64 的单图平均延迟更低、吞吐量更高，说明批处理摊薄了固定开销并提高了算子利用率
- `width_mult=0.5` 的 Batch=64 P95 为 44.373 ms，比 35.399 ms 的中位数高约 25.4%，相对抖动高于另外两个宽度；原始计时全部保留在 `latency_samples.csv`
- 本次结果只代表当前单线程 CPU 上的纯模型前向性能，不能直接推广到其他 CPU、GPU、线程数或完整应用链路
