import os
import torch
from torch import nn
from transformers import AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

from Dynamic_MoE.modeling.modeling_moe_adak import MoEForCausalLM

# torchrun --nproc_per_node=1 train_router.py
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

    def compute_loss(self, model, inputs):
        """
        1) 正常运行 MoE LLM forward（HF Trainer 会算 LM loss）
        2) 从模型中拉取 allocator 的 PPO buffer
        3) 按论文 Eq.(6) 计算 reward（最后一层 LM log-likelihood）
        4) 组装 PPO batches
        5) 执行 PPO 更新
        6) 返回 LM loss（用于 HF 训练管线）
        """

        # -------------------------
        # 0. 运行原始模型 forward
        # -------------------------
        outputs = model(**inputs)
        lm_loss = outputs.loss                     # HF Trainer 用这个做优化
        real_model = model.module if hasattr(model, "module") else model


        # -------------------------
        # 1. 按论文 Eq.(6) 计算 REWARD
        #
        #    R = log P(x_i | x_1,...,x_{i-1})   仅最后一层 allocator 使用
        #
        # -------------------------
        with torch.no_grad():
            logits = outputs.logits.float()        # [B, S, V]
            labels = inputs["labels"]              # [B, S]
            B, S = labels.shape

            # ---- Shift: LM 预测下一个 token ----
            shift_logits = logits[:, :-1]          # predict labels[:,1:]
            shift_labels = labels[:, 1:]

            log_probs = torch.log_softmax(shift_logits, dim=-1)
            token_logp = torch.gather(
                log_probs, -1, shift_labels.unsqueeze(-1)
            ).squeeze(-1)                          # [B, S-1]

            # ---- mask padding tokens ----
            token_logp = token_logp * (shift_labels != -100)

            # ---- 最后一个有效 token 的 logP ----
            reward = token_logp.clone()             # [B,S-1]
            reward = torch.clamp(reward, -50, 50)   # optional
            reward = torch.cat(
                [reward, torch.zeros(B,1, device=reward.device)],  # pad 最后一个位置
                dim=1
            )  # → [B,S]

            # # ---- PPO 稳定 clip ----
            # reward = torch.clamp(reward, -50, 50)



        # -------------------------
        # 2. 拉取 PPO buffer（来自每一层 EnhancedSwitchMLP）
        # -------------------------
        ppo_data = real_model.ppo_buffer
        if len(ppo_data) == 0:
            return lm_loss


        # -------------------------
        # 3. 计算 advantage（可选：标准化）
        # -------------------------
        advantage = reward - reward.mean(dim=1, keepdim=True)
        std = advantage.std(dim=1, keepdim=True) + 1e-6
        advantage = advantage / std     # [B,S]


        # -------------------------
        # 4. 构建 PPO batches（论文 Eq.(6)：仅最后一层有 reward）
        # -------------------------
        ppo_batches = []
        last_layer = real_model.num_layers - 1

        for item in ppo_data:
            layer_id = item["layer_id"]

            if layer_id == last_layer:
                adv = advantage          # 论文要求：最后一层=真实奖励
            else:
                adv = torch.zeros_like(advantage)   # 其它层=0

            ppo_batches.append({
                "layer_id": item["layer_id"],
                "state": item["state"],               # [B,S,H]
                "action": item["action"],             # [B,S]
                "old_log_prob": item["old_log_prob"], # [B,S]
                "advantage": adv,                     # [B,1]
                "old_alloc_logits": item["old_alloc_logits"],
            })


        # -------------------------
        # 5. PPO 更新 allocator（只更新 W_alloc）
        # -------------------------
        real_model.ppo_update(ppo_batches)
        real_model.clear_ppo_buffer()

        # 看一眼k分布
        layer_hists = []
        for layer_id, layer in enumerate(real_model.model.layers):
            if hasattr(layer.mlp, "k_hist"):
                layer_hists.append(layer.mlp.k_hist.detach().cpu())

                # reset histogram for next step
                layer.mlp.k_hist.zero_()

        if len(layer_hists) > 0:
            # sum across layers
            total_hist = torch.stack(layer_hists).sum(dim=0)

            # multi-GPU sync
            import torch.distributed as dist
            if dist.is_initialized():
                total_hist = total_hist.to('cuda')
                dist.all_reduce(total_hist, op=dist.ReduceOp.SUM)

            print(f"[Monitor] total k distribution = {total_hist.tolist()}")

        # -------------------------
        # 6. 返回 LM loss（不影响 PPO）
        # -------------------------
        return lm_loss



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
        learning_rate=1e-3,
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
