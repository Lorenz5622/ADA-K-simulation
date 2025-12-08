# ============================================================
# train_finetune.py — 解冻大模型主体 + allocator：标准微调训练脚本
# 直接运行： torchrun --nproc_per_node=1 train_finetune.py
# ============================================================

import os
import torch
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
from ADAK.modeling.modeling_moe_adak import MoEForCausalLM   # ★ 你的 MoE 模型

# ------------------------------------------------------------
# 1. 构建模型（全部参数可训练）
# ------------------------------------------------------------
def load_model(path):
    print("Loading MoE model from:", path)
    model = MoEForCausalLM.from_pretrained(path)

    # ★★ 关键：全部参数解冻（包括 LLM + Router + Experts + Allocator）
    for name, p in model.named_parameters():
        p.requires_grad = True

    print("Total trainable params:",
          sum(p.numel() for p in model.parameters() if p.requires_grad))

    return model


# ------------------------------------------------------------
# 2. 微调损失函数：标准 LM Loss
# ------------------------------------------------------------
class FinetuneTrainer(Trainer):
    """普通 SFT：LM Loss = CrossEntropyLoss"""
    def compute_loss(self, model, inputs, return_outputs=False):

        labels = inputs["labels"]
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels,                 # ★ 自动计算 CE Loss
            output_router_logits=False     # ★ 禁用 PPO 路由输出
        )

        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


# ------------------------------------------------------------
# 3. 数据处理（以 PIQA 为例）
# ------------------------------------------------------------
def load_piqa(tokenizer):
    ds = load_dataset("piqa", trust_remote_code=True)

    tokenizer.pad_token = tokenizer.unk_token

    def preprocess(batch):
        text = [g + " " + s for g, s in zip(batch["goal"], batch["sol1"])]
        out = tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True
        )
        out["labels"] = out["input_ids"].copy()
        return out

    ds = ds.map(preprocess, batched=True)
    ds.set_format(type="torch",
                  columns=["input_ids", "attention_mask", "labels"])
    return ds


# ------------------------------------------------------------
# 4. 主训练入口
# ------------------------------------------------------------
def main():

    BASE_PATH = "/data/cyx/models" if os.path.exists("/data") else "/root/models"
    MODEL_PATH = f"{BASE_PATH}/ADAK_MoE"    # 你的 MoE 初始模型
    SAVE_PATH  = f"{BASE_PATH}/ADAK_MoE_finetune"   # 微调后模型输出目录
    from transformers import BitsAndBytesConfig
    # 1) 加载模型
    model = MoEForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16
    )
    print("Trainable parameters:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # 2) 数据集
    dataset = load_piqa(tokenizer)

    # 3) TrainingArguments（标准微调超参）
    args = TrainingArguments(
        output_dir=SAVE_PATH,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,                     # ★ MoE 微调建议小 lr
        logging_steps=50,
        save_steps=2000,
        save_total_limit=2,
        bf16=True,
        ddp_find_unused_parameters=True,
    )

    # 4) Trainer
    trainer = FinetuneTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
    )

    print("==== Start Fine-tuning (Backbone + Allocator together) ====")
    trainer.train()

    # 5) 保存结果
    print("==== Saving model ====")
    trainer.save_model(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    print("Done.")


if __name__ == "__main__":
    main()