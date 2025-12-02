import os
import torch
from torch import nn
from transformers import AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

from Dynamic_MoE.modeling.modeling_moe_adak import MoEForCausalLM


# ============================================================
# 1. 载入模型（必须是你修改后的 allocator 版本）
# ============================================================

def load_model(path):
    print("Loading model:", path)
    model = MoEForCausalLM.from_pretrained(path)

    # 冻结所有参数
    for name, p in model.named_parameters():
        p.requires_grad = False

    # 解冻 allocator（论文 W_alloc）
    trainable_params = []
    for name, p in model.named_parameters():
        if "allocator" in name:     # ★ 只训练 W_alloc
            p.requires_grad = True
            trainable_params.append(p)

    print("Trainable allocator params:", sum(p.numel() for p in trainable_params))
    return model


# ============================================================
# 2. PPO Trainer（覆盖 HF backward）
# ============================================================

class PPOTrainer(Trainer):

    def training_step(self, model, inputs):
        """
        覆盖 Trainer 的 training_step，使 HF 不再 backward LM loss，
        PPO 更新在 compute_loss 中执行。
        """
        model.train()
        loss = self.compute_loss(model, inputs)

        # 不执行 HF backward（论文要求冻结 LLM）
        return loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        在这里执行：
        1）forward 得到 LM loss（作为 reward）
        2）收集 allocator 的 PPO trajectory
        3）执行 PPO 更新（仅 rank0）
        """
        real_model = model.module if hasattr(model, "module") else model

        # ===============================
        # 1. Forward：LLM loss 仅用作 reward
        # ===============================
        outputs = model(**inputs)
        lm_loss = outputs.loss  # 不反向，用于 reward

        # ===============================
        # 2. 取出 PPO buffer（来自 EnhancedSwitchMLP）
        # ===============================
        ppo_data = real_model.ppo_buffer
        real_model.ppo_buffer = []

        if len(ppo_data) > 0:
            # ===============================
            # 3. 从 logits 产生 reward（logP）
            # ===============================
            with torch.no_grad():
                logits = outputs.logits.float()
                labels = inputs["labels"]
                log_probs = torch.log_softmax(logits, dim=-1)
                token_logp = torch.gather(
                    log_probs, -1, labels.unsqueeze(-1)
                ).squeeze(-1)
                token_logp = token_logp * (labels != -100)
                reward = token_logp.mean(dim=-1)    # [B]

            # 安全 clip（避免 nan）
            reward = torch.clamp(reward, -50, 50)

            # ===============================
            # 4. 计算 advantage（归一化版）
            # ===============================
            advantage = reward - reward.mean()
            std = advantage.std()
            if std < 1e-6:
                std = torch.tensor(1.0, device=advantage.device)
            advantage = advantage / std
            advantage = advantage.unsqueeze(-1)  # [B,1]

            # ===============================
            # 5. 按 allocator 信息组 PPO batch
            # ===============================
            ppo_batches = []
            for item in ppo_data:
                ppo_batches.append({
                    "alloc_logits": item["alloc_logits"],
                    "alloc_probs": item["alloc_probs"],
                    "sampled_k":   item["sampled_k"],
                    "advantage":   advantage,   # reward signal
                })

            # ===============================
            # 6. PPO update（仅 rank0 执行）
            # ===============================
            import torch.distributed as dist

            if (not dist.is_initialized()) or dist.get_rank() == 0:
                real_model.ppo_update(ppo_batches)

            if dist.is_initialized():
                dist.barrier()

        # ===============================
        # 返回去梯度的 LM loss（HF 不会 backward）
        # ===============================
        return (lm_loss.detach(), outputs) if return_outputs else lm_loss.detach()


# ============================================================
# 3. 数据处理（PIQA）
# ============================================================

def load_piqa(tokenizer):
    print("Loading PIQA dataset...")
    ds = load_dataset("piqa", trust_remote_code=True)

    tokenizer.pad_token = tokenizer.unk_token

    def preprocess(batch):
        text = [g + " " + s for g, s in zip(batch["goal"], batch["sol1"])]
        out = tokenizer(text, padding="max_length",
                        truncation=True, max_length=128)
        out["labels"] = out["input_ids"].copy()
        return out

    ds = ds.map(preprocess, batched=True)
    ds.set_format(type="torch",
                  columns=["input_ids", "attention_mask", "labels"])
    return ds


# ============================================================
# 4. 训练入口
# ============================================================

def main():

    PATH_PREFIX = "/root"
    if os.path.exists("/home/cyx"):
        PATH_PREFIX = "/home/cyx"

    MODEL_PATH = f"{PATH_PREFIX}/models/Dynamic_MoE"
    SAVE_PATH = f"{PATH_PREFIX}/models/ADAK_MoE"

    # =============== 加载模型 ===============
    model = load_model(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # =============== 加载数据 ===============
    dataset = load_piqa(tokenizer)

    # =============== Trainer 设置 ===============
    training_args = TrainingArguments(
        output_dir=SAVE_PATH,
        learning_rate=1e-4,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        save_strategy="epoch",
        logging_steps=20,
        ddp_find_unused_parameters=False,
        bf16=True,          # 强烈推荐
        tf32=True,          # 进一步提速
    )

    # =============== PPO Trainer ===============
    trainer = PPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    model.save_pretrained(SAVE_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
