import ipex_llm
import torch
import torch.nn.functional as F

device = "xpu"
dtype = torch.float16
# device = "cpu"
# dtype = torch.float32

x = torch.randn((12000,2048), device=device, dtype=dtype) # * 57->368M
w1 = torch.randn((64,2048,704), device=device, dtype=dtype) # * 945M
w2 = torch.randn((64,352,2048), device=device, dtype=dtype) # * 994M
indices = torch.randint(0, 12000, (12000,), device=device, dtype=torch.int64) # * 999M
topk = 6
hidden_size = 2048
topk_weights = torch.randn((2000, 6), device=device, dtype=dtype) # * 1004M






#!>>>>>>>>>>>org>>>>>>>>>>>>
x = torch.matmul(x, w1) # ! occupy to 5050M, x.shape->(64,12000,704)
x = F.silu(x[..., :352]) * x[..., 352:] #! occupy to 5100M
x = torch.matmul(x, w2)# ! occupy to 8095M
#!<<<<<<<<<<<org<<<<<<<<<<<<

#*>>>>>>>>>>>june>>>>>>>>>>>
# batch_size = 8  # 每次计算 8 个 w1 的矩阵，控制显存占用
# output = torch.empty(64, 12000, 704, device=device) # ? 3057M
# for i in range(0, 64, batch_size):
#     batch_w1 = w1[i:i+batch_size]  
#     output[i:i+batch_size, :, :] = torch.matmul(x, batch_w1)
# x = output # ? 3457M
# x = F.silu(x[..., :352]) * x[..., 352:] # ? occupy to 5536M

# output = torch.empty(64, 12000, 2048, device=device)
# for i in range(0, 64, batch_size):
#     batch_w2 = w2[i:i+batch_size]  # 形状 (batch_size, 352, 2048)
#     output[i:i+batch_size, :, :] = torch.matmul(x, batch_w2)
# x = output # occupy to 3500M
#*<<<<<<<<<<<june<<<<<<<<<<<

# x.shape=(64,12000,2048), indices.shape=(12000)
#1179648000000 bytes
x = x[indices]
x = x.view(-1, topk, hidden_size)

x = x * topk_weights.unsqueeze_(dim=-1)
x = x.sum(dim=-2)
x = x.view(2000, 2048)