# %%
import numpy as np

def conv2d_manual(image, kernel):
    """
    image: (H, W) 的二维数组
    kernel: (kH, kW) 的二维数组
    """
    H, W = image.shape
    kH, kW = kernel.shape

    # 输出特征图的尺寸
    out_H = H - kH + 1
    out_W = W - kW + 1

    output = np.zeros((out_H, out_W))

    # 双重循环，滑动窗口
    for i in range(out_H):
        for j in range(out_W):
            # 取出当前窗口覆盖的图片区域
            region = image[i:i+kH, j:j+kW]
            # 对应元素相乘后求和，就是这个位置的卷积结果
            output[i, j] = np.sum(region * kernel)

    return output
# %%
image = np.array([
    [1, 2, 3, 0, 1],
    [0, 1, 2, 3, 0],
    [1, 0, 1, 2, 3],
    [2, 1, 0, 1, 2],
    [0, 2, 1, 0, 1]
], dtype=float)

kernel = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1]
], dtype=float)

result = conv2d_manual(image, kernel)
print("输出特征图形状:", result.shape)
print(result)

# %%
import torch
import torch.nn.functional as F

# PyTorch的conv2d要求输入是4维: (batch_size, channels, H, W)
# 我们现在只有1张图、1个通道，所以要把维度"凑"出来
image_torch = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
kernel_torch = torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

print("image_torch形状:", image_torch.shape)   # 应该是 (1,1,5,5)
print("kernel_torch形状:", kernel_torch.shape) # 应该是 (1,1,3,3)

result_torch = F.conv2d(image_torch, kernel_torch)
print("PyTorch卷积结果:")
print(result_torch.squeeze().numpy())   # squeeze()把多余的维度去掉，方便看

# %%
import torch.nn as nn

# 模拟一张CIFAR-10图片: (batch=1, channels=3, H=32, W=32)
fake_image = torch.randn(1, 3, 32, 32)

conv_layer = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)

output = conv_layer(fake_image)
print("输出特征图形状:", output.shape)

# 统计这一层的参数量
total_params = sum(p.numel() for p in conv_layer.parameters())
print("这一层的总参数量:", total_params)
# %%
# 对比三种设置的输出尺寸差异
conv_no_pad = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0)
conv_same_pad = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
conv_stride2 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)

x = torch.randn(1, 3, 32, 32)

print("无padding:", conv_no_pad(x).shape)
print("same padding:", conv_same_pad(x).shape)
print("stride=2:", conv_stride2(x).shape)
# %%
