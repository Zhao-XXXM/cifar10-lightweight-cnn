# %%
import torch
print(torch.__version__)
# %%
a = torch.tensor(2.0)              # 不加 requires_grad
b = torch.tensor(2.0, requires_grad=True)   # 加了 requires_grad

y1 = a * 3
y2 = b * 3

print(y1.grad_fn)   # 猜猜这里会打印什么
print(y2.grad_fn)   # 猜猜这里会打印什么
# %%
x = torch.tensor(2.0, requires_grad=True)
y = x * 3
z = y + 1

z.backward()          # 从z出发，沿着计算图反向传播
print(x.grad)          # 查看PyTorch自动算出来的 ∂z/∂x
# %%
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

c = a * b + a**2
c.backward()

print(a.grad)
print(b.grad)

# %%
x = torch.tensor(2.0, requires_grad=True)

y = x * 3
z = y + 1
z.backward()
print("第一次backward后:", x.grad)

x.grad.zero_()      # 手动清零，PyTorch的Tensor自带这个方法
print("清零后:", x.grad)

y = x * 3
z = y + 1
z.backward()
print("清零后再backward:", x.grad)

# %%
