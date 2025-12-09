import os
import math
import numpy as np
import torch
from torch import nn
from collections import defaultdict

from transformers import AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from ADAK.modeling.modeling_moe_adak import MoEForCausalLM


# torchrun --nproc_per_node=1 train_router.py
# ============================================================
# 1. 载入模型（冻结 LLM，只训练 actor_critic）
# ============================================================

def load_model(path):
    print("Loading model:", path)
    model = MoEForCausalLM.from_pretrained(path)

    # 冻结所有参数
    for _, p in model.named_parameters():
        p.requires_grad = False

    # 只解冻 actor_critic（PPO / warm-start 都只改它）
    trainable_params = []
    for name, p in model.named_parameters():
        if "actor_critic" in name:
            p.requires_grad = True
            trainable_params.append(p)

    print("Trainable actor_critic params:", sum(p.numel() for p in trainable_params))
    return model


# ============================================================
# 2. 工具函数：GAE（token 维度）
# ============================================================

@torch.no_grad()
def compute_gae_tokenwise(
    rewards: torch.Tensor,   # [B, S]
    values: torch.Tensor,    # [B, S]
    dones: torch.Tensor,     # [B, S]  1=terminal/pad, 0=continue
    gamma: float,
    lam: float,
):
    """
    返回：
      advantages: [B, S]
      returns:    [B, S]
    """
    B, S = rewards.shape
    # bootstrap 的 V(s_{S})=0（token 序列末尾）
    values_ext = torch.cat([values, torch.zeros((B, 1), device=values.device, dtype=values.dtype)], dim=1)  # [B, S+1]

    advantages = torch.zeros_like(rewards)
    gae = torch.zeros((B,), device=rewards.device, dtype=rewards.dtype)

    for t in reversed(range(S)):
        not_done = 1.0 - dones[:, t]
        delta = rewards[:, t] + gamma * values_ext[:, t + 1] * not_done - values_ext[:, t]
        gae = delta + gamma * lam * not_done * gae
        advantages[:, t] = gae

    returns = advantages + values
    return advantages, returns


def masked_normalize(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8):
    """
    x:    [N]
    mask: [N]  bool or 0/1
    """
    mask = mask.float()
    denom = mask.sum().clamp_min(1.0)
    mean = (x * mask).sum() / denom
    var = ((x - mean) ** 2 * mask).sum() / denom
    std = torch.sqrt(var + eps)
    return (x - mean) / std


def make_minibatches(data_dict, batch_size: int, epochs: int):
    """
    data_dict: tensors with same first dim N
    返回：list[batch_dict]
    """
    N = next(iter(data_dict.values())).shape[0]
    batches = []
    for _ in range(epochs):
        idx = torch.randperm(N, device=next(iter(data_dict.values())).device)
        for start in range(0, N, batch_size):
            bidx = idx[start:start + batch_size]
            batch = {k: v[bidx] for k, v in data_dict.items()}
            batches.append(batch)
    return batches


# ============================================================
# 3. Warm Start Trainer（监督：actor logits 拟合 pseudo-k）
# ============================================================

class WarmStartAllocatorTrainer(Trainer):
    """
    Warm Start：对所有层的 actor（actor_critic.actor 输出的 logits）做监督学习
    pseudo-k 来自冻结 router logits 的 top-p 采样。
    """

    def compute_loss(self, model, inputs):
        real_model = model.module if hasattr(model, "module") else model

        outputs = real_model(**inputs, output_router_logits=True)
        hidden_states_all = outputs["all_hidden_states_for_warm"]   # List[L] each [B,S,H]
        router_logits_all = outputs["all_router_logits_for_warm"]   # List[L] each [B,S,E]

        num_layers = real_model.num_layers
        total_loss = 0.0
        count = 0

        for layer_id in range(num_layers):
            h = hidden_states_all[layer_id]             # [B,S,H]
            router_logits = router_logits_all[layer_id] # [B,S,E]

            pseudo_k = real_model.model.layers[layer_id].mlp.compute_pseudo_k_top_p(
                router_logits, top_p=0.9, temperature=1.0
            )  # [B,S] in 1..K

            # actor logits（替代原 allocator）
            actor_logits, _, _ = real_model.model.layers[layer_id].mlp.actor_critic(h)  # [B,S,K]
            B, S, K = actor_logits.shape

            # CE labels: 0..K-1
            pseudo_k = torch.clamp(pseudo_k, 1, K) - 1

            loss = nn.CrossEntropyLoss()(
                actor_logits.reshape(B * S, K),
                pseudo_k.reshape(B * S)
            )
            total_loss += loss
            count += 1

        return total_loss / max(count, 1)


# ============================================================
# 4. PPO Trainer（方案 A：禁用 HF optimizer，只走 ppo_update）
# ============================================================

class PPOTrainer(Trainer):
    def __init__(self, *args, ppo_gamma=0.99, ppo_lam=0.95,
                 ppo_epochs=4, ppo_minibatch_size=1024,
                 layer_reward_decay=0.9,
                 clip_range=0.2, vf_coef=0.5, ent_coef=0.01, max_grad_norm=1.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_history = []

        # PPO 超参
        self.ppo_gamma = ppo_gamma
        self.ppo_lam = ppo_lam
        self.ppo_epochs = ppo_epochs
        self.ppo_minibatch_size = ppo_minibatch_size
        self.layer_reward_decay = layer_reward_decay

        # 这些会透传给 model.ppo_update
        self.clip_range = clip_range
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

    # -------- 方案 A：禁用 HF optimizer / scheduler（但 Trainer 要求它们存在） --------
    def create_optimizer(self):
        # Trainer 必须有一个“非空” optimizer，否则 torch.optim 会报:
        # ValueError: optimizer got an empty parameter list
        #
        # 这里给一个不会实际更新的 optimizer：
        # - 参数：requires_grad=True（通常只有 actor_critic）
        # - lr=0.0 & weight_decay=0.0
        params = [p for p in self.model.parameters() if p.requires_grad]
        if len(params) == 0:
            # 兜底：至少塞一个参数避免 empty list（lr=0 不会更新）
            params = [next(self.model.parameters())]
        self.optimizer = torch.optim.AdamW(params, lr=0.0, weight_decay=0.0)
        return self.optimizer

    def create_scheduler(self, num_training_steps, optimizer=None):
        # HF Trainer 的训练循环会无条件调用 self.lr_scheduler.step()，所以不能返回 None。
        class _NoOpLRScheduler:
            def step(self, *args, **kwargs):
                return
            def get_last_lr(self):
                return [0.0]
        self.lr_scheduler = _NoOpLRScheduler()
        return self.lr_scheduler

    def optimizer_step(self, *args, **kwargs):
        # 方案A：阻止 HF training loop 执行 optimizer.step()
        # PPO 更新只在 real_model.ppo_update() 内部进行
        return

    def training_step(self, model, inputs):
        # 覆盖 training_step：不做 HF backward（否则会对无 grad 的 loss 调 backward 报错）
        model.train()
        loss = self.compute_loss(model, inputs)
        return loss.detach()

    def compute_loss(self, model, inputs):
        """
        1) forward 产出 logits（用于 token-level reward）
        2) 从 real_model.ppo_buffer 取轨迹
        3) 对每层：reward -> (GAE advantages, returns)
        4) flatten + mask -> per-layer minibatches
        5) 调用 real_model.ppo_update(per_layer_batches, ...)
        6) clear buffer + 记录 reward
        """
        outputs = model(**inputs)
        lm_loss = outputs.loss
        real_model = model.module if hasattr(model, "module") else model

        # ---------- 1) token-level reward ----------
        with torch.no_grad():
            logits = outputs.logits.float()                # [B,S,V]
            labels = inputs["labels"]                      # [B,S]
            attention_mask = inputs.get("attention_mask", None)  # [B,S] (0/1)

            B, S = labels.shape

            # shift
            shift_logits = logits[:, :-1]                  # [B,S-1,V]
            shift_labels = labels[:, 1:]                   # [B,S-1]

            # safe gather：避免 -100 索引报错
            safe_shift_labels = shift_labels.clone()
            mask_shift = torch.ones_like(safe_shift_labels, dtype=torch.float32, device=safe_shift_labels.device)
            if (safe_shift_labels == -100).any():
                mask_shift = (safe_shift_labels != -100).float()
                safe_shift_labels = torch.where(
                    safe_shift_labels == -100,
                    torch.zeros_like(safe_shift_labels),
                    safe_shift_labels
                )

            log_probs = torch.log_softmax(shift_logits, dim=-1)  # [B,S-1,V]
            token_logp = torch.gather(log_probs, -1, safe_shift_labels.unsqueeze(-1)).squeeze(-1)  # [B,S-1]
            token_logp = token_logp * mask_shift

            # 使用 attention_mask 做更稳的 mask（如有）
            if attention_mask is not None:
                shift_attn = attention_mask[:, 1:].float()  # [B,S-1]
                token_logp = token_logp * shift_attn

            reward = torch.clamp(token_logp, -10, 10)
            reward = torch.nn.functional.pad(reward, pad=(0, 1), mode="constant", value=0.0)  # [B,S]

            print(f"[Reward] reward={reward.mean().item():.6f}")

        # ---------- 2) 拉取 PPO buffer ----------
        ppo_data = real_model.ppo_buffer
        if len(ppo_data) == 0:
            return lm_loss

        # ---------- 3) 构造 mask / dones（token 维度） ----------
        if "attention_mask" in inputs and inputs["attention_mask"] is not None:
            token_mask = inputs["attention_mask"].float()  # [B,S]
        else:
            # fallback：labels != -100 视为有效
            token_mask = (labels != -100).float()

        # dones：padding 位置 done=1；每条序列最后一个有效 token done=1
        dones = (1.0 - token_mask).clone()  # padding done=1
        with torch.no_grad():
            lengths = token_mask.sum(dim=1).long().clamp_min(1)  # [B]
            last_idx = (lengths - 1).clamp_min(0)               # [B]
            dones[torch.arange(B, device=dones.device), last_idx] = 1.0

        # ---------- 4) per-layer 聚合：flatten + mask ----------
        per_layer_flat = defaultdict(lambda: defaultdict(list))
        last_layer = real_model.num_layers - 1

        for item in ppo_data:
            layer_id = int(item["layer_id"])
            layer_distance = last_layer - layer_id
            layer_scale = (self.layer_reward_decay ** layer_distance)

            state = item["state"]                  # [B,S,H]
            action = item["action"].long()         # [B,S]
            old_logp = item["old_log_prob"]        # [B,S]
            values = item["critic_values"].squeeze(-1)  # [B,S]

            # reward 分层衰减
            rewards_layer = reward * layer_scale   # [B,S]
            rewards_layer = rewards_layer * token_mask

            # GAE
            adv, ret = compute_gae_tokenwise(
                rewards_layer, values, dones,
                gamma=self.ppo_gamma, lam=self.ppo_lam
            )

            # flatten 并过滤无效 token
            flat_mask = (token_mask.reshape(-1) > 0.5)
            state_f = state.reshape(-1, state.shape[-1])[flat_mask]
            action_f = action.reshape(-1)[flat_mask]
            old_logp_f = old_logp.reshape(-1)[flat_mask]
            adv_f = adv.reshape(-1)[flat_mask]
            ret_f = ret.reshape(-1)[flat_mask]

            # advantage normalize（仅对有效 token）
            adv_f = masked_normalize(adv_f, torch.ones_like(adv_f, dtype=torch.float32))

            per_layer_flat[layer_id]["states"].append(state_f)
            per_layer_flat[layer_id]["actions"].append(action_f)
            per_layer_flat[layer_id]["log_probs"].append(old_logp_f)
            per_layer_flat[layer_id]["advantages"].append(adv_f)
            per_layer_flat[layer_id]["returns"].append(ret_f)

        # concat per layer
        per_layer_batches = {}
        for layer_id, buckets in per_layer_flat.items():
            states = torch.cat(buckets["states"], dim=0)
            actions = torch.cat(buckets["actions"], dim=0)
            log_probs_old = torch.cat(buckets["log_probs"], dim=0)
            advantages = torch.cat(buckets["advantages"], dim=0)
            returns = torch.cat(buckets["returns"], dim=0)

            data_dict = {
                "states": states,
                "actions": actions,
                "log_probs": log_probs_old,
                "advantages": advantages,
                "returns": returns,
            }

            per_layer_batches[layer_id] = make_minibatches(
                data_dict,
                batch_size=self.ppo_minibatch_size,
                epochs=self.ppo_epochs
            )

        # ---------- 5) PPO 更新（只靠 model.ppo_update 内部更新） ----------
        real_model.ppo_update(
            per_layer_batches,
            clip_range=self.clip_range,
            vf_coef=self.vf_coef,
            ent_coef=self.ent_coef,
            max_grad_norm=self.max_grad_norm
        )

        # 清理 buffer，避免重复用旧数据
        real_model.clear_ppo_buffer()

        # 看一眼 k 分布（保持你原来的监控逻辑）
        layer_hists = []
        for _, layer in enumerate(real_model.model.layers):
            if hasattr(layer.mlp, "k_hist"):
                layer_hists.append(layer.mlp.k_hist.detach().cpu())
                layer.mlp.k_hist.zero_()

        if len(layer_hists) > 0:
            total_hist = torch.stack(layer_hists).sum(dim=0)

            import torch.distributed as dist
            if dist.is_initialized():
                total_hist = total_hist.to("cuda")
                dist.all_reduce(total_hist, op=dist.ReduceOp.SUM)

            print(f"[Monitor] total k distribution = {total_hist.tolist()}")

        # ---------- 6) 记录 reward 并返回 lm_loss（仅用于 HF pipeline 的 logging） ----------
        self.reward_history.append(reward.mean().item())
        return lm_loss

# ============================================================
# 5. 数据处理（PIQA）
# ============================================================

def load_piqa(tokenizer):
    print("Loading PIQA dataset...")
    ds = load_dataset("piqa", trust_remote_code=True)

    tokenizer.pad_token = tokenizer.unk_token

    def preprocess(batch):
        text = [g + " " + s for g, s in zip(batch["goal"], batch["sol1"])]
        out = tokenizer(text, padding="max_length", truncation=True, max_length=128)
        out["labels"] = out["input_ids"].copy()
        return out

    ds = ds.map(preprocess, batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds


def warm_start_dataset(dataset, ratio=0.1):
    ds = dataset["train"]
    n = int(len(ds) * ratio)
    return {"train": ds.select(range(n))}


# ============================================================
# 6. 训练入口
# ============================================================

def main():
    JUMP_WARM_START = False   # 是否跳过 Warm-Start 直接 PPO 训练
    PATH_PREFIX = "/root"
    if os.path.exists("/data"):
        PATH_PREFIX = "/data/cyx"

    MODEL_PATH = f"{PATH_PREFIX}/models/Dynamic_MoE"
    SAVE_PATH = f"{PATH_PREFIX}/models/ADAK_MoE"

    # =============== 加载模型 ===============
    model = load_model(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # =============== 加载数据 ===============
    dataset = load_piqa(tokenizer)
    warm_ds = warm_start_dataset(dataset, ratio=0.1)

    # ====== Warm-Start 训练参数（小 lr、短 epoch） ======
    if not JUMP_WARM_START:
        warm_args = TrainingArguments(
            output_dir=SAVE_PATH + "/warm",
            learning_rate=5e-4,
            num_train_epochs=1,
            per_device_train_batch_size=4,
            logging_steps=20,
            gradient_accumulation_steps=1,
            save_total_limit=1,
            bf16=True,
            ddp_find_unused_parameters=False,
        )
        warm_trainer = WarmStartAllocatorTrainer(
            model=model,
            args=warm_args,
            train_dataset=warm_ds["train"],
            tokenizer=tokenizer,
        )
        print("===== Warm Start Training =====")
        warm_trainer.train()
        print("===== Warm Start Done =====")
        warm_trainer.save_model(SAVE_PATH + "/warm_model")
        tokenizer.save_pretrained(SAVE_PATH + "/warm_model")

        if os.path.exists(SAVE_PATH + "/warm"):
            from pathlib import Path
            import shutil
            folder = Path(SAVE_PATH + "/warm")
            shutil.rmtree(folder)

        # 释放
        del warm_trainer
        del model
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        torch.cuda.synchronize()

    print("Loading warm-start model for PPO training...")
    model = load_model(SAVE_PATH + "/warm_model")
    tokenizer = AutoTokenizer.from_pretrained(SAVE_PATH + "/warm_model")

    training_args = TrainingArguments(
        output_dir=SAVE_PATH,
        # learning_rate 对 PPOTrainer 无意义（HF optimizer 被禁用），保留不影响
        learning_rate=5e-4,
        num_train_epochs=16,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        save_strategy="steps",
        logging_steps=5000,
        save_total_limit=1,
        ddp_find_unused_parameters=False,
        bf16=True,
        tf32=True,
    )

    # =============== PPO Trainer ===============
    trainer = PPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,

        # PPO 超参（可按需调）
        ppo_gamma=0.99,
        ppo_lam=0.95,
        ppo_epochs=4,
        ppo_minibatch_size=1024,
        layer_reward_decay=0.9,

        clip_range=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=1.0,
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    print("Done.")

    import json
    with open(os.path.join(SAVE_PATH, "reward_history.json"), "w") as f:
        json.dump(trainer.reward_history, f)


if __name__ == "__main__":
    main()
