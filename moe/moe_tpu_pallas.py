import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch.nn.functional as F
from torch_xla.experimental.custom_kernel import _histogram


def fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
) -> torch.Tensor:
# device = xm.xla_device()
    device = torch.device("cpu")
    dtype = torch.float32

    hidden_states = torch.randn((120,2048), device=device, dtype=dtype)
    w1 = torch.randn((64,704,2048), device=device, dtype=dtype)
    w2 = torch.randn((64,2048,352), device=device, dtype=dtype)
    gating_output = torch.randn((120, 64), device=device, dtype=dtype)
    topk = 6
    renormalize = True
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
    assert (num_tokens * topk) % 16 == 0, (
        "The Pallas GMM kernel requires num_tokens * topk to be a multiple of "
        f"16 but got {num_tokens * topk}")

    hidden_states = hidden_states.view(num_tokens, hidden_size)
    gating_output = gating_output.view(num_tokens, num_experts)
    topk_weights = gating_output.softmax(dim=-1, dtype=torch.float)
    topk_weights, topk_indices = topk_weights.topk(topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)


    print("topk_indices.shape -> ", topk_indices.shape)
    topk_indices = topk_indices.flatten()
    print("flatten(), topk_indices.shape -> ", topk_indices.shape)
    topk_argsort_indices = topk_indices.argsort()
    print("topk_argsort_indices.shape -> ", topk_argsort_indices.shape)
    topk_argsort_revert_indices = topk_argsort_indices.argsort()
    print("topk_argsort_revert_indices.shape -> ", topk_argsort_revert_indices.shape)
    token_indices = torch.arange(num_tokens,
                                    device=device).repeat_interleave(topk)
    print("token_indices.shape -> ", token_indices.shape)
    print("topk_argsort_indices.shape -> ", topk_argsort_indices.shape)
    token_indices = token_indices[topk_argsort_indices]
    group_sizes = _histogram(topk_indices.to(torch.int32), 0, num_experts - 1)

    print("token_indices.shape -> ", token_indices.shape)
    print("group_sizes -> ", group_sizes.shape, group_sizes)

# NOTE(woosuk): The GMM Pallas kernel requires a different weight layout
# from HF Transformers.
    w1 = w1.transpose(1, 2)
    w2 = w2.transpose(1, 2)
    print("w1.shape->", w1.shape, "w2.shape->", w2.shape)

    print("hidden_states.shape->", hidden_states.shape)
    x = hidden_states[token_indices]
    print("x.shape->", x.shape)
    x = torch.ops.xla.gmm(x, w1, group_sizes)
    print("gmm(x, w1) , x.shape->", x.shape)
    x = F.silu(x[..., :intermediate_size]) * x[..., intermediate_size:]
    print("silu , x.shape->", x.shape)
    x = torch.ops.xla.gmm(x, w2, group_sizes)
    print("gmm(x, w2) , x.shape->", x.shape)
    x = x[topk_argsort_revert_indices].reshape(-1, topk, hidden_size)
    print("x.shape->", x.shape)

    x = x * topk_weights.unsqueeze_(dim=-1)
    x = x.sum(dim=-2)
    x = x.reshape(orig_shape)
    return x



def set_tpu_seed(seed):
    torch.manual_seed(seed)
    xm.set_rng_state(seed)

if __name__ == '__main__':
    set_tpu_seed(42)
    device = xm.xla_device()
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