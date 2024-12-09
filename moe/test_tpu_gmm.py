import torch
import torch_xla
import torch_xla.core.xla_model as xm

if __name__=="__main__":
    lhs = torch.tensor([
        [1, 2],  # Group 1
        [3, 4],  # Group 1
        [5, 6],  # Group 2
        [7, 8]   # Group 2
    ], dtype=torch.float32)
    rhs = torch.tensor([
        [1, 0], 
        [0, 1]
    ], dtype=torch.float32)
    group_sizes = torch.tensor([2, 4])  # Group 1: Rows 0-1, Group 2: Rows 2-3

    print(torch.ops.xla.gmm(lhs, rhs, group_sizes))