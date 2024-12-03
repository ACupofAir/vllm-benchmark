import ipex_llm
import torch
import torch.nn.functional as F

#*>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>|Jun W. Code|>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def custom_histogram(indices, min, max):
    sorted_indices, _ = torch.sort(indices, descending=False)
    result = torch.zeros(max - min + 1, dtype=torch.int32, device=indices.device)
    # for i from min to max, get the first index of element in indices that is equal to i
    for i in range(min, max + 1):
        result[i - min] = (sorted_indices==i).sum()
    return result


def custom_gmm(x, w, group_sizes):
    group_num = group_sizes.shape[0]
    # x.shape=(*, dim),
    # w.shape=(group_num, dim, intermediate_size)
    # group_sizes.shape=(group_num)

    # for group i, use w[i] to calculate x
    results = []
    for i in range(group_num):
        st = group_sizes[:i].sum()
        ed = group_sizes[:i+1].sum()
        # print(st,ed,ed-st)
        y = torch.matmul(x[st:ed], w[i].squeeze())
        results.append(y)
    return torch.cat(results, dim=0)
#*<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<|Jun W. Code|<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
) -> torch.Tensor:
    """
    Args:
        hidden_states: [*, hidden_size]
        w1: [num_experts, intermediate_size * 2, hidden_size]
        w2: [num_experts, hidden_size, intermediate_size]
        gating_output: [*, num_experts]
    """
    orig_shape = hidden_states.shape
    hidden_size = hidden_states.shape[-1]
    num_tokens = hidden_states.shape[:-1].numel()
    num_experts = w1.shape[0]
    intermediate_size = w2.shape[-1]
    device = hidden_states.device
    dtype = hidden_states.dtype
    print('======================DEBUG START: numtoken, topk======================')
    print(num_tokens, topk)
    print('======================DEBUG  END : numtoken, topk======================')
    # assert (num_tokens * topk) % 16 == 0, (
    #     "The Pallas GMM kernel requires num_tokens * topk to be a multiple of "
    #     f"16 but got {num_tokens * topk}")

    hidden_states = hidden_states.view(num_tokens, hidden_size)
    gating_output = gating_output.view(num_tokens, num_experts)
    topk_weights = gating_output.softmax(dim=-1, dtype=torch.float)
    topk_weights, topk_indices = topk_weights.topk(topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    topk_indices = topk_indices.flatten()
    topk_argsort_indices = topk_indices.argsort()
    topk_argsort_revert_indices = topk_argsort_indices.argsort()
    token_indices = torch.arange(num_tokens,
                                 device=device).repeat_interleave(topk)
    token_indices = token_indices[topk_argsort_indices]
    group_sizes = custom_histogram(topk_indices.to(torch.int32), 0, num_experts - 1)

    # NOTE(woosuk): The GMM Pallas kernel requires a different weight layout
    # from HF Transformers.
    w1 = w1.transpose(1, 2)
    w2 = w2.transpose(1, 2)

    x = hidden_states[token_indices]
    x = custom_gmm(x, w1, group_sizes)
    # x = torch.ops.xla.gmm(x, w1, group_sizes)
    x = F.silu(x[..., :intermediate_size]) * x[..., intermediate_size:]
    x = custom_gmm(x, w2, group_sizes)
    # x = torch.ops.xla.gmm(x, w2, group_sizes)
    x = x[topk_argsort_revert_indices].reshape(-1, topk, hidden_size)

    x = x * topk_weights.unsqueeze_(dim=-1)
    x = x.sum(dim=-2)
    x = x.reshape(orig_shape)
    return x


def set_xpu_seed(seed):
    torch.manual_seed(seed)
    torch.xpu.manual_seed_all(seed)
    torch.xpu.manual_seed(seed)

if __name__ == '__main__':
    set_xpu_seed(42)
    device = "xpu"
    dtype = torch.float16
    input_len = 16
    num_experts = 64
    intermediate_size = 352
    hidden_size = 16
    hidden_states = torch.randn((input_len, hidden_size), device=device, dtype=dtype)
    w1 = torch.randn((num_experts, intermediate_size*2, hidden_size), device=device, dtype=dtype)
    w2 = torch.randn((num_experts, hidden_size, intermediate_size), device=device, dtype=dtype)
    gating_output = torch.randn((input_len, num_experts), device=device, dtype=dtype)
    topk = 6
    renormalize = True
    out = fused_moe(hidden_states, w1, w2, gating_output, topk, renormalize)
    print(out)
    print(out.shape)