#%%
import torch
import matplotlib.pyplot as plt

# 定义 Silu 激活函数
def silu(x):
    return x * torch.sigmoid(x)

# 生成输入数据
x = torch.linspace(-5, 5, 1000)  # 从 -5 到 5 的 1000 个点

# 计算 Silu 函数的输出
y = silu(x)

#%%
# 绘制图像
plt.figure(figsize=(8, 6))
plt.plot(x.numpy(), y.numpy(), label='Silu (Swish)', color='b', linewidth=2)
plt.title('Silu Activation Function (Swish)', fontsize=14)
plt.xlabel('Input', fontsize=12)
plt.ylabel('Output', fontsize=12)
plt.grid(True)
plt.legend()
plt.show()
# %%
plt.show()

#%%

import matplotlib.pyplot as plt

# 绘制图形
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])

# 手动显示图像
plt.show()
# %%
