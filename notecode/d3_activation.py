# %%
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 造训练数据：y = x^2，取-3到3之间的点
x_train = torch.linspace(-3, 3, 100).unsqueeze(1)   # 形状变成 (100, 1)
y_train = x_train ** 2

# 模型A：纯Linear堆叠，没有激活函数
model_linear_only = nn.Sequential(
    nn.Linear(1, 16),
    nn.Linear(16, 16),
    nn.Linear(16, 1)
)

# 模型B：带ReLU激活函数
model_with_relu = nn.Sequential(
    nn.Linear(1, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

print(model_linear_only)
print(model_with_relu)
# %%
def train_model(model, epochs=500, lr=0.01):
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_history = []

    for epoch in range(epochs):
        y_pred = model(x_train)
        loss = loss_fn(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

    return loss_history

# 分别训练两个模型
loss_A = train_model(model_linear_only)
loss_B = train_model(model_with_relu)

print(f"模型A（无激活函数）最终loss: {loss_A[-1]:.4f}")
print(f"模型B（带ReLU）最终loss:   {loss_B[-1]:.4f}")

# %%
model_linear_only.eval()   # 切换到评估模式
model_with_relu.eval()

with torch.no_grad():          # 只是画图看效果，不需要计算图，用昨天学的no_grad
    y_pred_A = model_linear_only(x_train)
    y_pred_B = model_with_relu(x_train)

plt.figure(figsize=(8, 5))
plt.plot(x_train.numpy(), y_train.numpy(), label="True: y=x^2", linewidth=2)
plt.plot(x_train.numpy(), y_pred_A.numpy(), label="Model A (no activation)", linestyle="--")
plt.plot(x_train.numpy(), y_pred_B.numpy(), label="Model B (with ReLU)", linestyle="--")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear-only vs ReLU: Fitting y=x^2")
plt.show()

# %%
