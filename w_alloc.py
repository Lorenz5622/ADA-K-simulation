from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer
import torch
from Dynamic_MoE.modeling.modeling_moe_adak import MoEForCausalLM
from Dynamic_MoE.modeling.configuration_moe import MoEConfig
import json
import torch.nn.functional as F
import random
from Dynamic_MoE.rl.rl import GeneticAlgorithm
import numpy as np
import os
from datetime import datetime

if os.path.exists("/home/cyx") :
    PATH_PREFIX = "/home/cyx"
model_path = f'{PATH_PREFIX}/models/Dynamic_MoE'
now = datetime.now().strftime("%Y%m%d_%H%M%S")  # 例如：20251007_213015
num_epochs = 100
# 构造文件名
record_file = f"../records/output_{now}.txt"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.unk_token

model_config = MoEConfig.from_pretrained(model_path,trust_remote_code=True)
model = MoEForCausalLM.from_pretrained(
    model_path,
    from_tf=False,
    config=model_config,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).cuda()
# 1. 冻结除 ExpertSelector 外的所有参数
for name, param in model.named_parameters():
    if "expert_selector.router" not in name:
        param.requires_grad = False
    else:
        param.requires_grad = True
        print(f"Will train parameter: {name}")

# 2. 验证设置
trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
print("Trainable parameters:")
for name in trainable_params:
    print(f"  {name}")

# 3. 设置优化器
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=1e-3
)

# 4. 标准训练循环
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        # 假设 batch 是一个字典，包含 input_ids, attention_mask 等
        outputs = model(**batch)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch} completed. Average Loss: {avg_loss}")

# 5. 保存训练后的 ExpertSelector 参数
model.save_pretrained("./expert_selector_trained")