import torch
import torch.nn as nn

# 1. 导入我们昨天编写的 BasicBlock
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# 2. 手写 ResNet-18 主干网络
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 64

        # Stem 层：适配 CIFAR-10 (32x32)，避免像 ImageNet 那样过早用 7x7 卷积和大 stride 丢失小图细节
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 4 个 Stage，每个 Stage 堆叠 2 个 BasicBlock
        self.layer1 = self._make_layer(out_channels=64,  num_blocks=2, stride=1)
        self.layer2 = self._make_layer(out_channels=128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(out_channels=256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(out_channels=512, num_blocks=2, stride=2)

        # 全局平均池化 + 分类器
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # 第一个 Block 可能降采样，后续 Block stride 均为 1
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride=s))
            self.in_channels = out_channels * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

if __name__ == '__main__':
    # 模拟 CIFAR-10 数据 (Batch Size=2, 3 通道, 32x32 图像)
    dummy_input = torch.randn(2, 3, 32, 32)
    model = ResNet18(num_classes=10)
    output = model(dummy_input)

    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"✅ ResNet-18 组装成功！")
    print(f"📦 总可训练参数量 (Params): {total_params:.2f} M (百万)")
    print(f"📐 输入尺寸: {dummy_input.shape} ---> 输出概率 Logits 尺寸: {output.shape}")