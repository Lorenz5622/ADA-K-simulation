# distributed_ga_moe.py
"""
Distributed GA evaluator for Dynamic_MoE (multi-GPU).
将每代的个体分配到各 GPU 并行计算 loss，rank0 聚合并执行 GA 操作。

使用方法:
    python distributed_ga_moe.py

请确保:
- CUDA_VISIBLE_DEVICES 能看到所有欲用 GPU
- PATH_PREFIX/models/Dynamic_MoE 存在
- datasets 下存在对应 DATASET (默认 "piqa") 的 train.jsonl & train-labels.lst
"""

import os
import json
import random
import numpy as np
from datetime import datetime
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

# --------- 你原来的 imports (保留) ----------
from transformers import AutoTokenizer
from Dynamic_MoE.modeling.modeling_moe import MoEForCausalLM
from Dynamic_MoE.modeling.configuration_moe import MoEConfig
from Dynamic_MoE.rl.rl import GeneticAlgorithm
import torch.nn.functional as F

# --------- 可配置项（你可以按需要修改） ----------
N_GENERATIONS = 500
CROSSOVER_RATE = 0.5
MUTATION_RATE = 0.08
MAX_DATASET_COUNT = 100
MAX_DATASET_EPOCHS = 70
INDIVIDUAL_COUNT = 28
DATASET = "piqa"
PATH_PREFIX = "/root"   # 自动切换为 /home/cyx 如果存在
WORLD_SIZE = 2
USE_EXIST_POP = False
EXIST_POP = [
    [3, 0, 3, 1, 3, 5, 2, 6, 3, 4, 1, 3, 6, 4, 3, 3, 4, 4, 5, 5, 2, 4, 3, 5],
    [2, 0, 2, 1, 3, 3, 2, 6, 3, 4, 1, 2, 5, 4, 4, 3, 4, 4, 5, 3, 4, 4, 3, 4],
    [3, 0, 3, 1, 3, 5, 2, 7, 3, 4, 1, 3, 6, 4, 3, 3, 4, 4, 5, 5, 2, 4, 3, 5],
    [3, 0, 3, 1, 3, 5, 2, 6, 3, 4, 1, 3, 6, 4, 3, 3, 4, 4, 5, 5, 2, 4, 3, 5],
    [2, 0, 2, 1, 3, 3, 2, 6, 3, 4, 1, 2, 5, 4, 4, 3, 4, 4, 5, 3, 4, 4, 3, 4],
    [3, 0, 3, 1, 3, 5, 2, 7, 3, 4, 1, 3, 6, 4, 3, 3, 4, 4, 5, 5, 2, 4, 3, 5],
    [2, 0, 2, 1, 4, 4, 2, 6, 3, 4, 1, 2, 5, 4, 4, 3, 4, 4, 5, 4, 4, 3, 3, 4],
    [2, 0, 1, 1, 4, 3, 2, 6, 3, 4, 1, 2, 4, 4, 4, 3, 4, 4, 5, 4, 5, 3, 4, 4],
    [0, 4, 1, 0, 0, 5, 1, 0, 1, 2, 0, 2, 5, 6, 6, 2, 1, 6, 0, 0, 0, 0, 5, 4],
    [2, 0, 2, 1, 3, 3, 2, 6, 3, 4, 1, 2, 5, 4, 4, 3, 4, 4, 5, 4, 4, 2, 3, 4],
    [4, 0, 2, 3, 0, 1, 5, 2, 4, 0, 5, 2, 4, 1, 6, 0, 5, 2, 1, 4, 4, 3, 4, 4],
    [3, 3, 2, 3, 3, 5, 1, 0, 1, 2, 0, 2, 5, 6, 7, 2, 1, 7, 0, 0, 0, 1, 5, 4],
    [0, 4, 1, 0, 0, 6, 4, 6, 6, 1, 0, 5, 0, 1, 6, 2, 2, 5, 0, 3, 5, 1, 5, 0],
    [1, 6, 1, 1, 5, 4, 5, 1, 4, 4, 5, 1, 2, 2, 2, 2, 2, 6, 4, 0, 3, 3, 4, 6],
    [3, 0, 3, 1, 3, 6, 2, 7, 3, 4, 2, 3, 5, 4, 3, 2, 4, 4, 5, 6, 2, 4, 3, 5],
    [2, 0, 2, 1, 3, 3, 1, 6, 3, 4, 1, 2, 4, 4, 4, 4, 4, 4, 5, 4, 4, 3, 3, 4],
    [3, 1, 3, 3, 5, 2, 7, 6, 4, 0, 5, 5, 6, 1, 5, 5, 2, 4, 6, 2, 4, 2, 5, 4],
    [6, 6, 4, 5, 2, 0, 5, 6, 5, 0, 6, 5, 7, 6, 7, 1, 4, 3, 1, 0, 6, 6, 1, 3],
    [5, 3, 5, 3, 6, 7, 4, 1, 0, 3, 7, 2, 5, 0, 6, 1, 4, 1, 0, 4, 5, 4, 5, 2],
    [1, 1, 4, 5, 3, 4, 2, 3, 1, 4, 1, 6, 5, 5, 1, 6, 7, 3, 0, 2, 6, 6, 5, 4],
    [2, 0, 3, 1, 4, 2, 2, 7, 3, 4, 1, 2, 5, 4, 4, 2, 4, 4, 5, 4, 4, 3, 3, 4],
    [3, 0, 3, 1, 3, 5, 2, 6, 2, 4, 1, 3, 6, 4, 3, 3, 4, 4, 5, 5, 3, 4, 3, 5],
    [2, 0, 3, 1, 4, 3, 2, 6, 3, 3, 1, 2, 5, 4, 4, 3, 4, 4, 5, 4, 4, 3, 3, 5],
    [3, 0, 3, 1, 3, 4, 2, 6, 3, 4, 2, 4, 5, 4, 3, 3, 5, 4, 5, 4, 4, 3, 4, 4],
    [2, 0, 2, 1, 3, 3, 2, 6, 3, 4, 1, 2, 5, 4, 4, 3, 4, 4, 5, 4, 4, 3, 3, 4],
    [2, 0, 3, 1, 2, 5, 2, 7, 2, 4, 1, 2, 5, 4, 4, 3, 4, 4, 4, 5, 4, 4, 3, 4],
    [3, 0, 3, 1, 3, 5, 2, 7, 3, 4, 1, 3, 5, 4, 3, 3, 5, 4, 5, 3, 4, 3, 4, 3],
    [3, 0, 3, 1, 2, 5, 2, 7, 3, 4, 1, 3, 5, 4, 2, 3, 3, 4, 5, 2, 3, 2, 4, 3]
]
if os.path.exists("/home/cyx"):
    PATH_PREFIX = "/home/cyx"

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")

# --------- 保留或轻微改动的函数（来自你原始脚本） ----------
def generate_batch(tokenizer, model, prompts, dynamic_k=None, device='cuda'):
    try:
        with torch.no_grad():
            tokens = tokenizer(prompts, return_tensors="pt", padding=True)
            input_ids = tokens.input_ids.to(device)
            
            generate_ids = model.generate(
                inputs=input_ids,
                num_beams=1, 
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                dynamic_k=dynamic_k,
                max_new_tokens=1,
                top_p=0.9, 
                temperature=1.0, 
                do_sample=True
            )
            
            outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
            # 提取生成部分（去除输入部分）
            responses = []
            response_ids = []
            for i, output in enumerate(outputs):
                input_length = input_ids.shape[1]  # 统一的输入长度
                response = output[input_length:]  # 跳过输入部分
                responses.append(response)
                response_ids.append(generate_ids[i, input_length:])  # 提取生成的token IDs
            
            # 把 response_ids 堆成 tensor（可能长短不一，返回 list）
            return responses, response_ids
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            torch.cuda.empty_cache()
            return [], []
        else:
            raise

def calculate_batch_loss(tokenizer, model, prompts, labels, isOOM, device='cuda'):
    if isOOM:
        return 1000.0
    try:
        with torch.no_grad():
            # saved_logits 在你的模型会被保存为 list，第一个元素是 logits
            outputs = model.saved_logits[0]  # [batch_size, seq_len, vocab_size]
            
            inputs = tokenizer(prompts, return_tensors="pt", padding=True)
            input_ids = inputs.input_ids.to(device)
            
            shift_logits = outputs[..., :-1, :].contiguous().float()
            shift_labels = input_ids[..., 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=tokenizer.pad_token_id)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1)).view(shift_labels.size())
            
            lens = (input_ids != tokenizer.pad_token_id).sum(-1).cpu().numpy()
            
            ce_loss = loss.float().sum(-1).cpu().detach().numpy() / lens
            return float(ce_loss.mean())
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            torch.cuda.empty_cache()
            return 1000.0
        else:
            raise

def find_pattern(output_ids):
    pattern_1 = [673, 29901]
    candidates = [29909, 29933, 29907]
    # output_ids assumed to be 1D tensor or list
    if isinstance(output_ids, torch.Tensor):
        arr = output_ids.view(-1).tolist()
    else:
        arr = list(output_ids)
    found_positions = []
    for i in range(len(arr) - len(pattern_1)):
        if arr[i:i+2] == pattern_1:
            next_token = arr[i+2]
            if next_token in candidates:
                found_positions.append(i+2)
                break
    return found_positions

def analysis_piqa(line_json, line_label):
    data = json.loads(line_json.strip())
    prompt = f"""
        Question: {data['goal']}
        Options:
        A: {data['sol1']}
        B: {data['sol2']}
        Answer:"""
    return prompt

# --------- 读取数据函数（每个进程都会调用，保证使用相同随机种子以得到一致顺序） ----------
def build_epochs_data(file_json, file_label, max_count, max_epochs, seed=1234):
    with open(file_json, 'r', encoding='utf-8') as f_json, \
         open(file_label, 'r', encoding='utf-8') as f_label:
        pairs = [
            (data_line, label_line)
            for data_line, label_line in zip(f_json, f_label)
            if data_line.strip() and label_line.strip()
        ]
    # 固定随机种子以保证所有进程得到相同的 shuffle & 切片
    random.seed(seed)
    random.shuffle(pairs)
    total_needed = max_count * max_epochs
    if len(pairs) < total_needed:
        raise ValueError(f"Dataset too small: need {total_needed} lines but got {len(pairs)}")
    pairs = pairs[:total_needed]
    epochs_data = []
    for epoch in range(max_epochs):
        start_idx = epoch * max_count
        end_idx = start_idx + max_count
        epoch_data = pairs[start_idx:end_idx]
        epochs_data.append(epoch_data)
    return epochs_data

# --------- 个体评估（在每个 GPU 上执行） ----------
def evaluate_individual_with_dataset(tokenizer, model, epochs_data, experts, device):
    # experts: 可为 list/np.array，表示每层选择的专家个数
    epoch_losses = []

    for epoch_data in epochs_data:
        prompts = []
        labels = []
        for line_json, line_label in epoch_data:
            prompt = analysis_piqa(line_json, line_label)
            prompts.append(prompt)
            labels.append(line_label.strip())

        # generate_batch 返回 responses list（或空）和 response_ids list
        responses, output_ids = generate_batch(tokenizer, model, prompts, dynamic_k=experts, device=device)
        isOOM = (len(responses) == 0)
        batch_loss = calculate_batch_loss(tokenizer, model, prompts, labels, isOOM, device=device)
        if batch_loss < 1000:
            epoch_losses.append(batch_loss)

        # 清理
        model.saved_logits = []
        model.collected_hidden_states = []
        if hasattr(model, 'past_key_values'):
            try:
                del model.past_key_values
            except Exception:
                pass
        torch.cuda.empty_cache()

    if len(epoch_losses) == 0:
        return 1000.0
    return float(np.mean(epoch_losses))

# --------- 分布式通信的工具（使用 all_gather_object） ----------
def all_gather_object_list(obj, world_size):
    # wrappers for dist.all_gather_object, returns list of objects of length world_size
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, obj)
    return gathered

# --------- worker 主函数（每个 GPU 一个进程） ----------
def worker_main(rank, world_size, args):
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
                            world_size=world_size, rank=rank)

    device = f'cuda:{rank}'
    model_path = f'{PATH_PREFIX}/models/Dynamic_MoE'

    # 加载 tokenizer + model（每进程独立）
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.unk_token

    model_config = MoEConfig.from_pretrained(model_path, trust_remote_code=True)
    model = MoEForCausalLM.from_pretrained(
        model_path,
        config=model_config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    # 不强制 torch.compile（看你环境是否稳定）
    # model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

    # GA 对象在所有进程都初始化，但只有 rank0 用于选择/交叉/变异
    ga = GeneticAlgorithm(INDIVIDUAL_COUNT, 24)
    if USE_EXIST_POP:
        ga.pop = EXIST_POP
    pop = ga.pop

    # 准备数据（每个进程按相同 seed 构造 epochs_data）
    file_json = f'{PATH_PREFIX}/datasets/{DATASET}/train.jsonl'
    file_label = f'{PATH_PREFIX}/datasets/{DATASET}/train-labels.lst'
    epochs_data = build_epochs_data(file_json, file_label, MAX_DATASET_COUNT, MAX_DATASET_EPOCHS, seed=1234)

    record_file = f"../records/output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    # 分代主循环
    for gen_count in range(N_GENERATIONS):
        # 1) rank0 计算 expert_list 并通过 all_gather 或 broadcast 让其他进程获得
        if rank == 0:
            expert_list = ga.translateDNA()  # list-like of length INDIVIDUAL_COUNT
        else:
            expert_list = None
        # broadcast expert_list by having rank0 pass and all gather the same object:
        # 使用 dist.broadcast_object_list 也可以，但 here 我们用 all_gather to simplify:
        if rank == 0:
            # rank0 keeps its expert_list; use all_gather to share same object with others by sending the object from rank0
            pass
        # We'll use dist.broadcast_object_list to broadcast single object (more direct)
        # Prepare a one-item list according to API
        ex_list_container = [expert_list]
        dist.broadcast_object_list(ex_list_container, src=0)
        expert_list = ex_list_container[0]

        # 2) 根据 world_size 将 expert_list 划分到每个 GPU
        total = len(expert_list)
        per_rank = (total + world_size - 1) // world_size
        start = rank * per_rank
        end = min((rank + 1) * per_rank, total)
        local_experts = expert_list[start:end]

        # 3) 本 rank 评估 local_experts
        local_losses = []
        oomCount_local = 0
        for idx, experts in enumerate(local_experts):
            try:
                # experts 可能为 numpy array 或其他结构，转换为 list 以便模型接收
                try:
                    dynk = experts.tolist()
                except Exception:
                    dynk = list(experts)
                print(f"rank: {rank} is processing individual {idx}")
                loss = evaluate_individual_with_dataset(tokenizer, model, epochs_data, dynk, device)
                local_losses.append(float(loss))
                if loss >= 1000:
                    oomCount_local += 1
            except Exception as e:
                # 捕获异常，避免单个个体崩溃全局流程
                print(f"[rank {rank}] exception evaluating individual idx {idx} : {e}")
                local_losses.append(1000.0)

        # 4) 全部进程收集 local_losses（长度可能不均） -> master 收集并扁平化
        gathered = all_gather_object_list(local_losses, world_size)  # list of lists from all ranks

        # 5) rank0 聚合并做 GA 操作
        if rank == 0:
            # flatten
            loss_list = []
            for part in gathered:
                if part:
                    loss_list.extend(part)
            # 如果长度不等于 INDIVIDUAL_COUNT，说明有人分配为空或有些个体未评估，填充 1000
            if len(loss_list) < INDIVIDUAL_COUNT:
                # pad
                loss_list.extend([1000.0] * (INDIVIDUAL_COUNT - len(loss_list)))
            loss_arr = np.array(loss_list[:INDIVIDUAL_COUNT])

            # 处理 OOM 判断（如果频繁 OOM，跳过该代）
            # 这里我们统计所有 ranks 的 oomCount
            total_ooms = 0
            # 必须先收集 oom counts from each rank; simpler: sum of values where loss==1000
            total_ooms = int((loss_arr >= 1000).sum())

            if total_ooms > INDIVIDUAL_COUNT * MAX_DATASET_EPOCHS / 4:
                with open(record_file, 'a') as f:
                    f.write(f"jump this iteration {gen_count} due to frequent CUDA OOM\n")
                print("jump this iteration due to frequent CUDA OOM")
                # 仍需继续 GA? 原代码是 continue -> 生成新一代? 它是跳过 fitness update & select
                # 这里我们跳过 selection 直接广播 current population unchanged
                # (no further operations)
            else:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[rank 0] generation No. {gen_count}, time {now}: losses mean {loss_arr.mean():.4f}")
                fitness = ga.get_fitness(loss_arr)
                ga.print_info()
                if gen_count % 2 == 0:
                    ga.write_to_record(record_file, gen_count)
                pop = ga.select(
                    elite_rate = 0.1,
                    diversity_threshold = 0.1,
                    cross_rate = CROSSOVER_RATE,
                    mutation_rate = MUTATION_RATE
                )
        # 6) 同步到下一代前 barrier
        dist.barrier()

    # 结束
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("Finished all generations.")

# --------- 启动入口 ----------
def main():
    world_size = WORLD_SIZE
    if world_size == 0:
        raise RuntimeError("No CUDA devices found. Please set CUDA_VISIBLE_DEVICES to GPUs you wish to use.")
    print(f"Starting distributed GA with {world_size} GPUs.")
    # spawn processes
    mp.spawn(worker_main, args=(world_size, None), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
