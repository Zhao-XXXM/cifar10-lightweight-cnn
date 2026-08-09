# 数据增强（Data Augmentation）
- 随机裁剪（RandomCrop）+ 随机水平翻转（RandomHorizontalFlip），让模型每次看到的图像都不一样，彻底解决过拟合
  
# 学习率余弦退火（Cosine Annealing LR Scheduler）
- 训练前期保持高学习率快速收敛，后期逐渐将学习率平滑降低到近乎 0，帮助模型精细拟合最优解