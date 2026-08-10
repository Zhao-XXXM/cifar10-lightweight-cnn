import torch
import torch.nn as nn


def standard_conv_params(in_channels, out_channels, kernel_size, bias=False):
    weight_params = in_channels * out_channels * kernel_size * kernel_size
    bias_params = out_channels if bias else 0
    return weight_params + bias_params


def depthwise_separable_params(in_channels, out_channels, kernel_size, bias=False):
    depthwise_params = in_channels * kernel_size * kernel_size
    pointwise_params = in_channels * out_channels
    bias_params = out_channels if bias else 0
    return depthwise_params + pointwise_params + bias_params


class StandardConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        # 逐通道空间卷积：每个输入通道单独做卷积
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.dw_bn = nn.BatchNorm2d(in_channels)

        # 1x1 卷积：负责通道融合
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.pw_bn = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.dw_bn(out)
        out = self.relu(out)

        out = self.pointwise(out)
        out = self.pw_bn(out)
        out = self.relu(out)
        return out


def count_trainable_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


if __name__ == "__main__":
    batch = torch.randn(2, 64, 32, 32)

    standard_conv = StandardConvBlock(64, 128, kernel_size=3, stride=1, padding=1)
    sep_conv = DepthwiseSeparableConv(64, 128, kernel_size=3, stride=1, padding=1)

    standard_out = standard_conv(batch)
    sep_out = sep_conv(batch)

    print("Day19: Depthwise Separable Convolution")
    print("=" * 44)

    print("[手算公式验证]")
    print(f"普通卷积参数量: {standard_conv_params(64, 128, 3, bias=False):,}")
    print(f"深度可分离卷积参数量: {depthwise_separable_params(64, 128, 3, bias=False):,}")

    print("\n[PyTorch 实际统计]")
    print(f"普通卷积层参数量: {count_trainable_params(standard_conv):,}")
    print(f"深度可分离卷积模块参数量: {count_trainable_params(sep_conv):,}")

    print("\n[形状验证]")
    print(f"输入尺寸: {batch.shape}")
    print(f"普通卷积输出尺寸: {standard_out.shape}")
    print(f"深度可分离卷积输出尺寸: {sep_out.shape}")

    reduction = count_trainable_params(standard_conv) / count_trainable_params(sep_conv)
    print(f"\n参数压缩倍数: {reduction:.2f}x")
