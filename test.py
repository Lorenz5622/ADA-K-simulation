import torch
logits_list = torch.load("saved_logits.pt")

# 打印信息
print(f"Total generated steps: {len(logits_list)}")
for i, logits in enumerate(logits_list):
    print(f"Step {i+1}: logits shape = {logits.shape}")