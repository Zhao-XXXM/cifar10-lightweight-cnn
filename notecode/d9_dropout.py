import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 1. 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 2. 网络结构：在全连接层前引入 Dropout
class SimpleCNN_BN_Dropout(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层 1
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        # 卷积层 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        # Dropout 正则化层：以 p=0.5 的概率随机丢失神经元
        self.dropout = nn.Dropout(p=0.5)

        # 全连接分类头
        self.fc = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)              # 展平为 (Batch, 2048)
        
        # 在送入全连接层决策之前，施加 Dropout
        x = self.dropout(x)
        x = self.fc(x)
        return x

model = SimpleCNN_BN_Dropout()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. 训练 1 个 Epoch
print("🚀 [SimpleCNN + BN + Dropout] 开始训练 1 个 Epoch...")
model.train()  # 注意：train() 模式下 Dropout 会生效！

for step, (images, labels) in enumerate(train_loader):
    outputs = model(images)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 4. 在测试集上评估 Top-1 准确率
print("🔍 评估模型性能...")
model.eval()   # 注意：eval() 模式下 Dropout 会自动失效！

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100.0 * correct / total
print(f"🎯 [加了 Dropout 后] CIFAR-10 测试集 Top-1 准确率: {accuracy:.2f}%")