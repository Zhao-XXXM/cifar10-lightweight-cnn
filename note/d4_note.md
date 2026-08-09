## 今日学习目标
- 手写二维卷积，理解卷积核滑动和局部加权求和
- 用 PyTorch 的 `F.conv2d` 验证手写卷积结果
- 理解 `nn.Conv2d` 的输入输出形状、参数量、padding 和 stride

## 卷积
- 卷积就是用一个小窗口在图片上滑动
- 每滑到一个位置，就取出窗口覆盖的局部区域，与卷积核逐元素相乘再求和
- 手写 `conv2d_manual(image, kernel)` 可以帮助理解卷积不是神秘操作，本质就是局部矩阵运算

## PyTorch 卷积输入格式
- `F.conv2d` 要求输入图像是 4 维：`(batch_size, channels, H, W)`
- 单张单通道图片需要通过 `unsqueeze(0).unsqueeze(0)` 补成 `(1, 1, H, W)`
- 卷积核也需要补成 4 维：`(out_channels, in_channels, kH, kW)`

## 多通道卷积原理
- 对 RGB 图像来说，输入通道数是 3
- 一个卷积核的形状是 `(in_channels, k, k)`，会同时覆盖所有输入通道
- 每个通道分别做局部乘加，最后把所有通道结果相加，得到一张输出特征图
- 有多少个 `out_channels`，就有多少个卷积核，也就输出多少张特征图

## 参数量公式
- 普通卷积参数量：`out_channels * in_channels * k * k + out_channels`
- 最后一项 `out_channels` 来自 bias
- 例如 `Conv2d(3, 32, kernel_size=3)` 的参数量为：`32 * 3 * 3 * 3 + 32 = 896`

## 输出尺寸公式
- 输出高度：`out_H = (H + 2 * padding - kernel_size) / stride + 1`
- 输出宽度同理
- 实际计算结果必须是整数，否则卷积窗口无法整齐滑动

## padding
- padding 是在输入图像边缘补 0
- `kernel_size=3, stride=1, padding=1` 时，可以让 32x32 输入保持 32x32 输出
- padding 可以减少边缘信息丢失，也是 CNN 中很常见的设置

## stride
- stride 表示卷积核每次滑动的步长
- `stride=2` 会让特征图尺寸大约减半
- stride 可以用于降采样，减少后续层的计算量

## 卷积相比全连接层的优势
- 局部连接：每个卷积核只关注局部区域，适合提取边缘、纹理等局部视觉特征
- 权值共享：同一个卷积核在整张图上复用，极大减少参数量
- 保留空间结构：卷积输出仍然是特征图，不会像全连接那样直接打散空间位置

## 今日自查问题
- 为什么 CIFAR-10 图片输入到 Conv2d 时通道数是 3？
- `padding=1` 为什么能让 3 * 3 卷积保持尺寸不变？
- `out_channels=32` 代表输出 32 个类别吗？
