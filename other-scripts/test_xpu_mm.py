import ipex_llm
import torch
import torch.nn.functional as F

# device = "xpu"
# dtype = torch.float16
device = "cpu"
dtype = torch.float32

x = torch.randn((12000,2048), device=device, dtype=dtype)
w1 = torch.randn((64,2048,704), device=device, dtype=dtype)
w2 = torch.randn((64,352,2048), device=device, dtype=dtype)
indices = torch.randint(0, 12000, (12000,), device=device, dtype=torch.int64)
topk = 6
hidden_size = 2048
topk_weights = torch.randn((2000, 6), device=device, dtype=dtype)



x = torch.matmul(x, w1)
x = F.silu(x[..., :352]) * x[..., 352:]
x = torch.matmul(x, w2)
# x.shape=(64,12000,2048), indices.shape=(12000)
#1179648000000 bytes
x = x[indices]
x = x.view(-1, topk, hidden_size)

x = x * topk_weights.unsqueeze_(dim=-1)
x = x.sum(dim=-2)
x = x.view(2000, 2048)