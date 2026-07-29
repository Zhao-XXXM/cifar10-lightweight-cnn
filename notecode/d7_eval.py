import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 1. 数据预处理（训练集与测试集使用相同的标准化参数）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 加载训练集与测试集
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 2. 模型定义 (SimpleCNN)
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. 训练 1 个 Epoch
print("🚀 开始训练 1 个 Epoch...")
model.train()  # 开启训练模式

for step, (images, labels) in enumerate(train_loader):
    outputs = model(images)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 4. 在测试集上评估 Top-1 准确率
print("🔍 训练完成，开始在测试集上评估模型性能...")
model.eval()   # 开启评估模式

correct = 0    # 统计预测正确的图片总数
total = 0      # 统计测试集图片总数

# 评估阶段强行关闭梯度计算，省显存并加速
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)  # outputs 形状为 (batch_size, 10)

        # 取每个样本 10 个 Logits 中得分最大的那一个类别的索引（即预测类别）
        _, predicted = torch.max(outputs, dim=1)

        total += labels.size(0)                    # 累加当前 Batch 的样本数（64）
        correct += (predicted == labels).sum().item() # 统计预测与真实标签相符的数量

# 计算最终准确率
accuracy = 100.0 * correct / total
print(f"🎯 CIFAR-10 测试集 Top-1 准确率: {accuracy:.2f}%")