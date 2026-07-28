import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 1. 数据准备：加载 CIFAR-10 并打包成 DataLoader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # 归一化
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# 2. 搭建 SimpleCNN 模型（复用 Day 5 的网络结构）
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(32 * 8 * 8, 10)   # 2048 -> 10 维分类输出

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)              # 展平特征图为一维向量
        x = self.fc(x)
        return x

model = SimpleCNN()

# 3. 定义损失函数与优化器
criterion = nn.CrossEntropyLoss()               # 交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器，学习率 0.001

# 4. 开始训练（先跑 1 个 Epoch 验证代码完整性）
print("🚀 开始训练 1 个 Epoch...")
model.train()                                   # 切换为训练模式

for step, (images, labels) in enumerate(train_loader):
    # 4.1 前向传播：把一个 Batch 的图片塞给模型，得到预测结果
    outputs = model(images)
    loss = criterion(outputs, labels)

    # 4.2 反向传播与参数更新
    optimizer.zero_grad()                       # 1. 梯度清零
    loss.backward()                             # 2. 反向求导
    optimizer.step()                            # 3. 更新参数

    # 每隔 100 个 Batch（即处理了 6400 张图）打印一次当前的 Loss 变化
    if (step + 1) % 100 == 0:
        print(f"Step [{step + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

print("✅ Day 6 训练顺利跑通！")