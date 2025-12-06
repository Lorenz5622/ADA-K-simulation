import torch
router_logits = torch.rand(1, 10, 16)  # 均匀分布 U(0,1)
sampled_k = torch.randint(low=0, high=6, size=(1, 10))
# router_B, router_S, router_E = router_logits.shape
# indices = torch.arange(router_E, device=router_logits.device).repeat(router_B, router_S, 1)

# mask = torch.ones(router_B,router_S,router_E, dtype=torch.bool)
# print(f"sampled_k: {sampled_k}")
# print(f"sample_k.shape: {sampled_k.shape}")
# range_E = torch.arange(router_E, device=router_logits.device)[None,None,:]
# k_mask = range_E < sampled_k.unsqueeze(-1)

# # 选择 expert
# topk_idx = torch.where(k_mask, indices, -1)
# topk_scores = torch.where(k_mask, router_logits, 0.0)
print(router_logits)
print(sampled_k)
B, S, E = router_logits.shape
logits = router_logits.clone()
max_k = sampled_k.max().item()

# 保存 top-k 轮次选中 expert index，长度 max_k，每个是 [B,S]
all_indices = []
active_mask = sampled_k > 0

for _ in range(max_k):
    # 当前最大 expert id
    idx = logits.argmax(dim=-1)  # [B, S]

    idx = torch.where(active_mask, idx, torch.full_like(idx, -1))
    all_indices.append(idx)

    # 屏蔽被选中的 expert
    safe = idx.clone()
    safe[safe == -1] = 0
    logits.scatter_(dim=-1, index=safe.unsqueeze(-1), value=float('-inf'))

    sampled_k -= 1
    active_mask = sampled_k > 0

# all_indices → [B,S,max_k]
topk_indices = torch.stack(all_indices, dim=-1)

# ---- 你要求的对位格式 ----
# 初始化为 -1
topk_indices_full = torch.full(
    (B, S, E),
    fill_value=-1,
    device=router_logits.device,
    dtype=torch.long
)

# 对于每一轮，将 selected expert index 写到对应位置
for i in range(max_k):
    idx_i = topk_indices[..., i]  # [B,S]
    valid = idx_i != -1

    if valid.any():
        # scatter 的 index 必须转成 [B,S,1]
        safe_idx = idx_i.clone()
        safe_idx[~valid] = 0

        topk_indices_full.scatter_(
            dim=-1,
            index=safe_idx.unsqueeze(-1),
            src=safe_idx.unsqueeze(-1)   # expert_id 对位写入
        )

# ---- selected_scores: 对位填概率 ----
selected_scores = torch.where(
    topk_indices_full != -1,
    router_logits,
    torch.zeros_like(router_logits)
)
print(topk_indices_full)
print(selected_scores)
