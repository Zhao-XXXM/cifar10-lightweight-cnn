import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 1. 独立定义一个可复用的 VGG 基础卷积块（VGG Block）
class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 使用 Sequential 容器将“2组 3x3卷积”打包
        self.block = nn.Sequential(
            # 第1个 3x3 卷积
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # 第2个 3x3 卷积
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # 2x2 最大池化降采样
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.block(x)

# 2. 组装 VGG-Slim 主干网络
class VGGSlim(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 特征提取主干 (Feature Extractor)
        self.features = nn.Sequential(
            VGGBlock(in_channels=3, out_channels=32),    # (3, 32, 32) -> (32, 16, 16)
            VGGBlock(in_channels=32, out_channels=64),   # (32, 16, 16) -> (64, 8, 8)
            VGGBlock(in_channels=64, out_channels=128)   # (64, 8, 8) -> (128, 4, 4)
        )
        
        # 分类头 (Classifier)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 3. 编写模型统计与 1 个 Epoch 验证代码
if __name__ == '__main__':
    model = VGGSlim()
    
    # 算力指标计算：自动统计模型的总可训练参数量 (Params)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 [VGG-Slim] 总可训练参数量 (Params): {total_params / 1e6:.2f} M (百万)")

    # 简单跑通 1 个 Epoch
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("🚀 开始训练 1 个 Epoch (观察 VGG-Slim 初步收敛情况)...")
    model.train()
    for step, (images, labels) in enumerate(train_loader):
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"🎯 [VGG-Slim Baseline] 1 个 Epoch 测试集准确率: {100.0 * correct / total:.2f}%")