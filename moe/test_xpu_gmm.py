import ipex_llm
import torch

def custom_gmm(x, w, group_sizes):
    results = []
    st = 0
    for i, ed in enumerate(group_sizes):
        group_x = x[st:ed, :]
        results.append(group_x @ w[i].squeeze())
        st = ed
    return torch.cat(results, dim=0)

if __name__=="__main__":

    lhs = torch.tensor([
        [1, 2], 
        [3, 4], 
        [5, 6], 
        [7, 8]  
    ], dtype=torch.float32)  

    rhs = torch.tensor([
        [[1, 0], [0, 1]],  
        [[2, 0], [0, 3]]   
    ], dtype=torch.float32)

    group_sizes = torch.tensor([2, 4], dtype=torch.int32)  

    print(custom_gmm(lhs, rhs, group_sizes))