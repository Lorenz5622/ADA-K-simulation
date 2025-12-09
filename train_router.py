import os
import torch
from torch import nn
from transformers import AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from datasets import concatenate_datasets
from ADAK.modeling.modeling_moe_adak import MoEForCausalLM

# torchrun --nproc_per_node=1 train_router.py
# ============================================================
# 1. 载入模型（必须是你修改后的 allocator 版本）
# ============================================================


# TODO 加一个ACTOR-CRITIC
# 李宏毅PPO
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
class WarmStartAllocatorTrainer(Trainer):
    """Warm Start 阶段：对所有层的 allocator 做监督学习（由 Router Top-P 生成 pseudo-k）"""

    def compute_loss(self, model, inputs):
        real_model = model.module if hasattr(model, "module") else model

        # ============================================================
        # 1) 前向：让 MoEModel 返回所有 hidden_states + router_logits（所有层）
        # ============================================================
        outputs = real_model(**inputs, output_router_logits=True)

        hidden_states_all = outputs["all_hidden_states_for_warm"]     # List[L] 每层 [B,S,H]
        router_logits_all = outputs["all_router_logits_for_warm"]     # List[L] 每层 [B,S,E]

        num_layers = real_model.num_layers
        total_loss = 0.0
        count = 0

        # ============================================================
        # 2) 逐层进行 warm-start 训练
        # ============================================================
        for layer_id in range(num_layers):

            h = hidden_states_all[layer_id]                 # [B,S,H]
            router_logits = router_logits_all[layer_id]     # [B,S,E]

            # --------------------------------------------------------
            # 2a. 使用你自定义 top-p 函数计算 pseudo-k
            # --------------------------------------------------------
            pseudo_k = real_model.model.layers[layer_id].mlp.compute_pseudo_k_top_p(
                router_logits,
                top_p=0.9,
                temperature=1.0,
            )  # [B,S]
            

            # --------------------------------------------------------
            # 2b. 前向 allocator 得到 alloc_logits
            # --------------------------------------------------------
            alloc_logits, _ = real_model.model.layers[layer_id].mlp.allocator(h)  # [B,S,K]
            B, S, K = alloc_logits.shape

            # --- 修复方案：把 pseudo_k 映射到合法标签范围 ---
            # 1) 限制为 1…max_k
            pseudo_k = torch.clamp(pseudo_k, 1, K)

            # 2) CE Loss 标签必须是 0…K-1，因此 shift 到 0-based
            pseudo_k = pseudo_k - 1
            # print(f"pseudo_k: {pseudo_k}")
            # --------------------------------------------------------
            # 2c. 计算 CrossEntropy loss
            # --------------------------------------------------------
            loss = nn.CrossEntropyLoss()(
                alloc_logits.reshape(B * S, K),
                pseudo_k.reshape(B * S)
            )

            total_loss += loss
            count += 1

        # ============================================================
        # 3) 多层 loss 平均（论文中所有层应均等权）
        # ============================================================
        total_loss = total_loss / count

        return total_loss
class PPOTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_history = []

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
            reward = torch.clamp(token_logp, -10, 10)
            # TODO pad是否正确？
            reward = torch.nn.functional.pad(reward, pad=(0, 1), mode="constant", value=0.0)
            

            print(f"[Reward] {reward.mean().item()} ")

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
        std = advantage.std(dim=1, keepdim=True)
        advantage = advantage / std     # [B,S]


        # -------------------------
        # 4. 构建 PPO batches（论文 Eq.(6)：仅最后一层有 reward）
        # -------------------------
        ppo_batches = []
        last_layer = real_model.num_layers - 1
        for idx, item in enumerate(ppo_data):
            layer_id = item["layer_id"]

            # if layer_id == last_layer:
            #     adv = advantage          # 论文要求：最后一层=真实奖励
            # else:
            #     adv = torch.zeros_like(advantage)   # 其它层=0
            layer_distance = last_layer - layer_id
            adv = advantage * (0.9 ** layer_distance)
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
        self.reward_history.append(reward.mean().item())
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
def warm_start_dataset(dataset, ratio=0.1):
    ds = dataset["train"]
    n = int(len(ds) * ratio)
    return {"train": ds.select(range(n))}

# ============================================================
# 4. 训练入口
# ============================================================
def repeat_dataset(ds, times=5):
    return {
        "train": torch.utils.data.ConcatDataset([ds["train"]] * times),
        "validation": torch.utils.data.ConcatDataset([ds["validation"]] * times)
    }
def main():

    PATH_PREFIX = "/root"
    # if os.path.exists("/home/cyx"):
    #     PATH_PREFIX = "/home/cyx"
    if os.path.exists("/data"):
        PATH_PREFIX = "/data/cyx"

    MODEL_PATH = f"{PATH_PREFIX}/models/Dynamic_MoE"
    SAVE_PATH = f"{PATH_PREFIX}/models/ADAK_MoE"

    # =============== 加载模型 ===============
    model = load_model(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # =============== 加载数据 ===============
    # ds1 = load_dataset("piqa")["train"]
    # ds2 = load_dataset("hellaswag")["train"]
    # ds3 = load_dataset("commonsense_qa")["train"]
    # all_train = concatenate_datasets([ds1, ds2, ds3])
    # dataset = {"train": all_train}
    dataset = load_piqa(tokenizer)
    # dataset = repeat_dataset(dataset, times=1)
    warm_ds = warm_start_dataset(dataset, ratio=0.1)
    # ====== Warm-Start 训练参数（小 lr、短 epoch） ======
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
    # =============== Trainer 设置 ===============

    #----------------释放模型
    del warm_trainer
    del model
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    torch.cuda.synchronize()

    print("Loading warm-start model for PPO training...")
    model = load_model(SAVE_PATH + "/warm_model")
    tokenizer = AutoTokenizer.from_pretrained(SAVE_PATH + "/warm_model")
    # TODO 改大num_train_epochs
    training_args = TrainingArguments(
        output_dir=SAVE_PATH,
        learning_rate=5e-4,
        num_train_epochs=15,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        save_strategy="steps",
        logging_steps=5000,
        save_total_limit=1,
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
    tokenizer.save_pretrained(SAVE_PATH)
    print("Done.")

    import json
    with open(os.path.join(SAVE_PATH, "reward_history.json"), "w") as f:
        json.dump(trainer.reward_history, f)


if __name__ == "__main__":
    main()
