# %%
import torch
import torch.nn as nn

# 一个简单的4x4特征图
x = torch.tensor([[
    [1.0, 3.0, 2.0, 4.0],
    [5.0, 6.0, 1.0, 2.0],
    [7.0, 2.0, 4.0, 3.0],
    [1.0, 0.0, 5.0, 8.0]
]]).unsqueeze(0)   # 凑成 (batch=1, channel=1, H=4, W=4)

print("原始特征图:")
print(x.squeeze())

maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
output = maxpool(x)

print("池化后:")
print(output.squeeze())
# %%
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一组：卷积+激活+池化
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二组
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 分类头
        self.fc = nn.Linear(32 * 8 * 8, 10)   

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)    # 展平：保留batch维度，其余全部拉成一维
        x = self.fc(x)
        return x

model = SimpleCNN()
print(model)

# %%
import torchvision
import torchvision.transforms as transforms

# 定义数据预处理：转成Tensor
transform = transforms.ToTensor()

# 下载CIFAR-10数据集
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)

print("训练集样本总数:", len(train_dataset))

# 取第一张图片看看
image, label = train_dataset[0]
print("图片形状:", image.shape)
print("标签(数字):", label)

classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
print("标签(类别名):", classes[label])

# %%
