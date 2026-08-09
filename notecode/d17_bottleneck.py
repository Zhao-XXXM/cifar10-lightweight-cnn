import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    # 🌟 瓶颈块的通道扩张倍数为 4！
    expansion = 4

    def __init__(self, in_channels, mid_channels, stride=1):
        super().__init__()
        out_channels = mid_channels * self.expansion

        # 1. 1x1 降维卷积：将通道数从 in_channels 压缩到 mid_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        # 2. 3x3 核心特征提取卷积（如果 stride=2，则在此处降采样）
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        # 3. 1x1 升维卷积：将通道数从 mid_channels 拓展回 out_channels (4 * mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # 旁路/快捷分支 (Shortcut Path)
        self.shortcut = nn.Sequential()
        # 如果维度不一致（尺寸不同或通道数不同），用 1x1 卷积调整维度
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        # 加上 shortcut 残差连接
        out += self.shortcut(x)
        out = self.relu(out)
        return out


if __name__ == '__main__':
    # 测试场景：模拟 ResNet-50 中 Stage 2 的输入 (Batch=2, Channel=256, Size=56x56)
    x = torch.randn(2, 256, 56, 56)
    
    # 实例化一个 Bottleneck，中间通道数设为 128，输出通道数自动变为 128 * 4 = 512
    bottleneck_block = Bottleneck(in_channels=256, mid_channels=128, stride=2)
    out = bottleneck_block(x)

    print(f"✅ Bottleneck 结构测试成功！")
    print(f"📐 输入尺寸: {x.shape}")
    print(f"📐 输出尺寸: {out.shape}")
    
    # 计算该 Bottleneck 的参数量
    params = sum(p.numel() for p in bottleneck_block.parameters() if p.requires_grad)
    print(f"📦 单个 Bottleneck 参数量: {params:,} 个")