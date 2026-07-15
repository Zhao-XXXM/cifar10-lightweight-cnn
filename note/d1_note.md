# Day 1

## 今日学习目标
- 理解Tensor与numpy数组的区别
- 理解计算图（computational graph）的构建与反向传播原理
- 搭建VS Code + Python虚拟环境开发环境

## 核心知识点

### 1. requires_grad 的作用
- `requires_grad=True` 的Tensor会被记录进计算图
- 默认不记录，是为了推理阶段节省内存/计算资源

### 2. 计算图与反向传播
- 反向传播本质：从终点沿图反向走，链式法则逐层相乘
- 梯度只保存在"叶子节点"，存在 `.grad` 属性中

## 环境记录
- PyTorch版本：2.13.0+cpu