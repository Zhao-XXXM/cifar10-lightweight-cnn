# %%
import torch

W = torch.randn(3, 2, requires_grad=True)   # 权重矩阵，3行2列
b = torch.zeros(2, requires_grad=True)       # 偏置，初始为0

x = torch.tensor([1.0, 2.0, 3.0])            # 一个样本，3个特征

y = x @ W + b     # @ 是矩阵乘法，等价于 x.matmul(W)
print("手写Linear层输出:", y)
print("W的形状:", W.shape)
# %%
import torch.nn as nn

linear_layer = nn.Linear(in_features=3, out_features=2)

x = torch.tensor([1.0, 2.0, 3.0])
y = linear_layer(x)

print("PyTorch Linear输出:", y)
print("自动生成的W:", linear_layer.weight)
print("自动生成的b:", linear_layer.bias)

# %%
import torch
import torch.nn as nn

# 造训练数据：y = 2x + 1
x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]])   # 4个样本，每个样本1个特征
y_train = torch.tensor([[3.0], [5.0], [7.0], [9.0]])   # 对应的正确答案

# 创建一个最简单的线性层：输入1维，输出1维
model = nn.Linear(1, 1)

print("训练前的参数:")
print("W =", model.weight.item())
print("b =", model.bias.item())

# 用模型当前的(随机)参数做一次预测，看看效果多差
y_pred = model(x_train)
print("训练前的预测:", y_pred.squeeze().tolist())
print("正确答案:      ", y_train.squeeze().tolist())

# %%
import torch.optim as optim

# 重新创建一个全新的模型（避免受刚才实验的影响）
model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()                          # 均方误差损失函数
optimizer = optim.SGD(model.parameters(), lr=0.01)   # 优化器：随机梯度下降，学习率0.01

for epoch in range(100):
    y_pred = model(x_train)              # 第1步：前向传播，算出当前预测
    loss = loss_fn(y_pred, y_train)       # 第2步：算预测值和正确答案的差距(loss)

    optimizer.zero_grad()                 # 第3步：清零上一轮遗留的梯度
    loss.backward()                       # 第4步：反向传播，自动算出W、b该往哪调
    optimizer.step()                      # 第5步：真正去调整W、b

    if epoch % 20 == 0:                   # 每20轮打印一次，观察loss变化趋势
        print(f"epoch {epoch}, loss = {loss.item():.4f}")

print("\n训练后的参数:")
print("W =", model.weight.item(), "  (目标是2.0)")
print("b =", model.bias.item(), "  (目标是1.0)")

# %%
import torch.optim as optim

model_buggy = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model_buggy.parameters(), lr=0.01)

for epoch in range(100):
    y_pred = model_buggy(x_train)
    loss = loss_fn(y_pred, y_train)

    # 注意:故意不写 optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"epoch {epoch}, loss = {loss.item():.4f}")

# %%
import matplotlib.pyplot as plt

# 重新完整训练一次"正确版"，这次把每一轮的loss都记下来，不再是每20轮才打印
model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

loss_history = []   # 用来存每一轮的loss值

for epoch in range(100):
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())   # .item()把tensor转成普通数字存起来

plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()

# %%
