import ipex_llm
import torch

a1 = torch.zeros((600, 1000, 1000), device="xpu:0")
a1 = torch.zeros((600, 1000, 1000), device="xpu:0")
torch.xpu.empty_cache()
a1 = torch.zeros((600, 1000, 1000), device="xpu:0") + 2
torch.xpu.empty_cache()
print(torch.xpu.memory_allocated())

a2 = torch.zeros((600, 1000, 1000), device="xpu:0")
a3 = torch.zeros((600, 1000, 1000), device="xpu:0")
a4 = torch.zeros((600, 1000, 1000), device="xpu:0")
a5 = torch.zeros((600, 1000, 1000), device="xpu:0")

torch.xpu.empty_cache()
print(torch.xpu.memory_allocated())
