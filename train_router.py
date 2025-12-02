import torch
from torch import nn
from transformers import AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import os
# -------------------------
# 1. 加载你自己的模型
# -------------------------
from Dynamic_MoE.modeling.modeling_moe_adak import MoEForCausalLM

class PPOTrainer(Trainer):
    def training_step(self, model, inputs):
        model.train()

        # ---- 正常调用 compute_loss（内部会 PPO update）----
        loss = self.compute_loss(model, inputs)

        # ---- 不调用 backward() ----
        # 直接返回 loss 值（仅用于 logging）
        return loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False):
        # --- unwrap for DDP ---
        real_model = model.module if hasattr(model, "module") else model

        # ====== 1. 正常前向 ======
        outputs = model(**inputs)
        lm_loss = outputs.loss

        # ====== 2. 取 PPO buffer ======
        ppo_data = real_model.ppo_buffer
        real_model.ppo_buffer = []

        if len(ppo_data) > 0:

            # ====== 3. reward = logP ======
            with torch.no_grad():
                logits = outputs.logits
                labels = inputs["labels"]
                log_probs = torch.log_softmax(logits, dim=-1)
                token_logp = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)
                token_logp = token_logp * (labels != -100)

            # ====== 4. advantage ======
            advantage = token_logp.mean(dim=-1, keepdim=True)
            advantage = advantage - advantage.mean()

            std = advantage.std()
            if std < 1e-6:             # 防止 batch 内 reward 全相等
                std = torch.tensor(1.0, device=advantage.device)

            advantage = advantage / std

            # ====== 5. 组 PPO batch ======
            ppo_batches = []
            for item in ppo_data:
                ppo_batches.append({
                    "state": item["state"],
                    "action": item["action"],
                    "log_prob": item["log_prob"],
                    "reward": advantage.detach(),
                    "advantage": advantage.detach(),
                })

            # ====== 6. PPO update only rank0 ======
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                real_model.ppo_update(ppo_batches)
            if dist.is_initialized():
                dist.barrier()

        # ====== 7. 返回不参与 backward 的 loss ======
        return (lm_loss.detach(), outputs) if return_outputs else lm_loss.detach()


PATH_PREFIX = "/root"   # 自动切换为 /home/cyx 如果存在
if os.path.exists("/home/cyx"):
    PATH_PREFIX = "/home/cyx"
MODEL_PATH = f'{PATH_PREFIX}/models/Dynamic_MoE'
SAVE_PATH  = f'{PATH_PREFIX}/models/ADAK_MoE'

def main():
    print("Loading model...")
    model = MoEForCausalLM.from_pretrained(MODEL_PATH)

    # -------------------------
    # 2. 冻结所有参数
    # -------------------------
    print("Freezing all model parameters except router...")

    for name, param in model.named_parameters():
        param.requires_grad = False

    # -------------------------
    # 3. 找到所有 router 参数并解冻
    # -------------------------
    trainable_params = []
    for name, param in model.named_parameters():
        if "expert_selector.router" in name:  # 你的 ExpertSelector
            param.requires_grad = True
            trainable_params.append(param)

    print(f"Trainable router params: {sum(p.numel() for p in trainable_params)}")
    print(f"Total params: {sum(p.numel() for p in model.parameters())}")

    # -------------------------
    # 4. 加载 PIQA 数据集
    # -------------------------
    print("Loading PIQA dataset...")
    dataset = load_dataset("piqa", trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.unk_token

    def preprocess(batch):
        # 把 goal + sol1 作为训练输入（你也可以设计别的任务）
        text = [g + " " + s for g, s in zip(batch["goal"], batch["sol1"])]
        out = tokenizer(text, truncation=True, padding="max_length", max_length=128)
        out["labels"] = out["input_ids"].copy()
        return out

    dataset = dataset.map(preprocess, batched=True)
    dataset.set_format(type="torch",
                    columns=["input_ids", "attention_mask", "labels"])

    # -------------------------
    # 5. 训练设置（只更新 router）
    # -------------------------
    training_args = TrainingArguments(
        output_dir=SAVE_PATH,
        learning_rate=1e-4,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        save_strategy="epoch",
        logging_steps=20,
        ddp_find_unused_parameters=False,
    )

    # -------------------------
    # 6. Trainer（自动梯度下降，只更新 router）
    # -------------------------
    trainer = PPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
    )

    print("Starting training...")
    trainer.train()

    # -------------------------
    # 7. 保存完整模型（路由器会一起保存）
    # -------------------------
    print(f"Saving model to {SAVE_PATH} ...")
    model.save_pretrained(SAVE_PATH)
    print("Done!")


if __name__ == "__main__":
    main()
