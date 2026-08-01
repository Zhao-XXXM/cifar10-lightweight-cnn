import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 1. 继承 Day 10 的 VGGSlim 模型结构
class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    def forward(self, x):
        return self.block(x)

class VGGSlim(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            VGGBlock(3, 32),
            VGGBlock(32, 64),
            VGGBlock(64, 128)
        )
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

# 2. 训练与验证逻辑封装
def main():
    # 设备选择（如果有 GPU 优先用 CUDA，没有则用 CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚡ 当前运行设备: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = VGGSlim().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练超参数设置
    EPOCHS = 10
    best_acc = 0.0  # 记录历史最高验证集准确率
    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)  # 创建权重保存目录

    print(f"\n🚀 开始 {EPOCHS} 个 Epoch 的完整训练...")
    
    for epoch in range(1, EPOCHS + 1):
        # ------------------- 训练阶段 -------------------
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, dim=1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = 100.0 * correct_train / total_train

        # ------------------- 验证阶段 -------------------
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, dim=1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = 100.0 * correct_val / total_val

        # 打印当前 Epoch 的详细日志
        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

        # ------------------- 保存最佳模型判断 -------------------
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_path = os.path.join(save_dir, "vgg_slim_best.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"  🔥 突破历史记录！最佳模型已保存至: {best_model_path} (Val Acc: {best_acc:.2f}%)")

    print(f"\n🎉 训练结束！全过程最高验证集准确率: {best_acc:.2f}%")

if __name__ == '__main__':
    main()