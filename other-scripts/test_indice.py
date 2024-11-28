import ipex_llm
import torch
device = "xpu"
arr = [1,2,3,4,5,6,7,8,9,10]
torch_arr = torch.tensor(arr)
indices = torch.randint(0,10,(10,), device=device, dtype=torch.int64)
torch_arr = torch_arr[indices]
print(torch_arr)