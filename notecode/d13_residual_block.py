import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    # BasicBlock 的通道扩展倍数为 1 (Bottleneck 会是 4)
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # 主分支 (Residual Path - F(x))
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 旁路/快捷分支 (Shortcut Path - x)
        self.shortcut = nn.Sequential()

        # 关键细节：如果输入输出尺寸或通道数不一致（比如 stride=2 进行降采样），
        # 快捷分支必须使用 1x1 卷积调整维度，才能与 F(x) 相加！
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # 1. 计算残差分支 F(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 2. 加上旁路分支 x：H(x) = F(x) + shortcut(x)
        out += self.shortcut(x)

        # 3. 最后再通过激活函数
        out = self.relu(out)
        return out


if __name__ == '__main__':
    # 测试场景 1：输入输出维度完全一致 (Identity Mapping)
    x1 = torch.randn(2, 64, 32, 32)
    block1 = BasicBlock(in_channels=64, out_channels=64, stride=1)
    out1 = block1(x1)
    print(f"✅ 场景 1（同维度相加）输入: {x1.shape} ---> 输出: {out1.shape}")

    # 测试场景 2：通道数翻倍且特征图降采样 (Projection Shortcut)
    x2 = torch.randn(2, 64, 32, 32)
    block2 = BasicBlock(in_channels=64, out_channels=128, stride=2)
    out2 = block2(x2)
    print(f"✅ 场景 2（降采样维度对齐）输入: {x2.shape} ---> 输出: {out2.shape}")