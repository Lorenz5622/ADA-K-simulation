# infer.py
import os
import json
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import datetime
from transformers import AutoTokenizer

from Dynamic_MoE.modeling.modeling_moe import MoEForCausalLM
from Dynamic_MoE.modeling.configuration_moe import MoEConfig
from Dynamic_MoE.rl.rl import GeneticAlgorithm

# 超参数定义
N_GENERATIONS = 500
CROSSOVER_RATE = 0.4
MUTATION_RATE = 0.05
MAX_DATASET_COUNT = 50
MAX_DATASET_EPOCHS = 120
INDIVIDUAL_COUNT = 30

def generate(tokenizer, model, text, dynamic_k=None):
    inputs = [text]
    tokens = tokenizer(inputs, return_tensors="pt")
    input_ids = tokens.input_ids.cuda()
    generate_ids = model.generate(inputs=input_ids,
                num_beams=1, 
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                dynamic_k=dynamic_k,
                max_new_tokens=1, top_p=0.9, temperature=1.0, do_sample=True)
    outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    response = [outputs[i][len(inputs[i]):] for i in range(len(outputs))][0]
    response_ids = generate_ids[:, input_ids.shape[1]:]
    return response, response_ids

def generate_batch(tokenizer, model, prompts, dynamic_k=None):
    try:
        with torch.no_grad():
            tokens = tokenizer(prompts, return_tensors="pt", padding=True)
            input_ids = tokens.input_ids.cuda()
            
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
            
            return responses, torch.stack(response_ids)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return [], torch.empty(0)  # 返回空结果

def calculate_batch_loss(tokenizer, model, prompts, labels, isOOM):
    if isOOM:
        return 1000
    try:
        with torch.no_grad():
            # 获取模型输出
            outputs = model.saved_logits[0]  # [batch_size, seq_len, vocab_size]
            
            # 批量tokenize输入
            inputs = tokenizer(prompts, return_tensors="pt", padding=True)
            input_ids = inputs.input_ids.cuda()
            
            # 构造 shift_logits 和 shift_labels
            shift_logits = outputs[..., :-1, :].contiguous().float()
            shift_labels = input_ids[..., 1:].contiguous()
            
            # 计算 loss
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=tokenizer.pad_token_id)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1)).view(shift_labels.size())
            
            # 计算有效长度
            lens = (input_ids != tokenizer.pad_token_id).sum(-1).cpu().numpy()
            
            # 计算平均 loss
            ce_loss = loss.float().sum(-1).cpu().detach().numpy() / lens
            return ce_loss.mean()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return 1000
    
def find_pattern(output_ids):
    # 定义目标模式
    pattern_1 = [673, 29901]
    candidates = [29909, 29933, 29907]

    # 转换为 tensor 方便处理
    output_ids = output_ids.view(-1)  # 确保是一维

    found_positions = []

    for i in range(len(output_ids) - len(pattern_1)):
        # 检查是否匹配前两个 token: [673, 29901]
        if output_ids[i:i+2].tolist() == pattern_1:
            # 检查下一个 token 是否是候选之一
            next_token = output_ids[i+2].item()
            if next_token in candidates:
                found_positions.append(i+2)  # 记录位置和具体哪个 token
                break

    return found_positions

def analysis_siqa(line_json, line_label):
    data = json.loads(line_json.strip())
    prompt = f"""
        Context: {data['context']}
        Question: {data['question']}
        Options:
        A: {data['answerA']}
        B: {data['answerB']}
        C: {data['answerC']}
        Answer:"""
    return prompt

def analysis_piqa(line_json, line_label):
    data = json.loads(line_json.strip())
    prompt = f"""
        Question: {data['goal']}
        Options:
        A: {data['sol1']}
        B: {data['sol2']}
        Answer:"""
    return prompt

def random_n_dataset(file_json, file_label):
    output_data = '/home/cyx/datasets/piqa/sampled_30_data.jsonl'
    output_label = '/home/cyx/datasets/piqa/sampled_30_labels.lst'

    # 读取数据和标签，并构建成 pairs
    with open(file_json, 'r', encoding='utf-8') as f_json, \
        open(file_label, 'r', encoding='utf-8') as f_label:

        pairs = [
            (json.loads(data_line.strip()), label_line.strip())
            for data_line, label_line in zip(f_json, f_label)
            if data_line.strip() and label_line.strip()
        ]
    sampled_pairs = random.sample(pairs, MAX_DATASET_COUNT*MAX_DATASET_EPOCHS)

    with open(output_data, 'w', encoding='utf-8') as f_data, \
        open(output_label, 'w', encoding='utf-8') as f_label:

        for data, label in sampled_pairs:
            # 保存原始数据（不加 label 字段）
            f_data.write(json.dumps(data, ensure_ascii=False) + '\n')
            # 保存标签（每行一个）
            f_label.write(label + '\n')
    
    return output_data, output_label

def evaluate_experts(model, tokenizer, experts_chunk, data_pairs, device):
    """评估一组专家配置"""
    loss_list = []
    oom_count = 0
    
    for i, experts in enumerate(experts_chunk):
        epoch_losses = []
        
        # 将数据分成多个epoch
        epochs_data = []
        for epoch in range(MAX_DATASET_EPOCHS):
            start_idx = epoch * MAX_DATASET_COUNT
            end_idx = start_idx + MAX_DATASET_COUNT
            epoch_data = data_pairs[start_idx:end_idx]
            epochs_data.append(epoch_data)
        
        for epoch_data in epochs_data:
            prompts = []
            labels = []

            for line_json, line_label in epoch_data:
                prompt = analysis_piqa(line_json, line_label)
                prompts.append(prompt)
                labels.append(line_label.strip())
            
            # 批量生成
            responses, output_ids = generate_batch(tokenizer, model, prompts, experts.tolist())
            
            # 批量计算loss
            isOOM = False
            if len(responses) == 0:
                isOOM = True
                oom_count += 1
            batch_loss = calculate_batch_loss(tokenizer, model, prompts, labels, isOOM)
            if batch_loss < 1000:
                epoch_losses.append(batch_loss)
            
            # 清理缓存
            model.saved_logits = []
            model.collected_hidden_states = []
            if hasattr(model, 'past_key_values'):
                del model.past_key_values
            torch.cuda.empty_cache()

        if len(epoch_losses) == 0:
            loss_list.append(1000)
        else:
            loss_list.append(np.mean(epoch_losses))
        
        print(f"device {device.index} loss is: {epoch_losses}")
    return loss_list, oom_count

def setup_model_on_gpu(gpu_id, model_path):
    """
    在指定GPU上加载模型
    """
    # 设置设备
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(gpu_id)
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.unk_token

    # 加载模型配置和模型
    model_config = MoEConfig.from_pretrained(model_path, trust_remote_code=True)
    model = MoEForCausalLM.from_pretrained(
        model_path,
        from_tf=False,
        config=model_config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(device)
    
    model.eval()
    return model, tokenizer, device

def main_worker(rank, world_size):
    """每个GPU上的工作进程"""
    # 初始化分布式环境
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # 设置当前设备
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    model_path = '/home/cyx/models/Dynamic_MoE'
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    record_file = f"output_{now}.txt"
    
    # 在当前GPU上加载模型
    model, tokenizer, _ = setup_model_on_gpu(rank, model_path)
    
    file_json = '/home/cyx/datasets/piqa/train.jsonl'
    file_label = '/home/cyx/datasets/piqa/train-labels.lst'
    file_json, file_label = random_n_dataset(file_json, file_label)
    
    # 只在主进程中初始化遗传算法
    if rank == 0:
        ga = GeneticAlgorithm(INDIVIDUAL_COUNT, 24)
        print(f"--------------begin iterations at {now} ---------------------")
    
    # 读取数据集
    with open(file_json, 'r', encoding='utf-8') as fj, \
         open(file_label, 'r', encoding='utf-8') as fl:
        data_pairs = list(zip(fj.readlines(), fl.readlines()))
        # 打乱顺序
        random.shuffle(data_pairs)
        # 限制处理的数据量
        data_pairs = data_pairs[:MAX_DATASET_COUNT*MAX_DATASET_EPOCHS]
    
    for gen_count in range(N_GENERATIONS):
        # 主进程生成新种群并分配任务
        expert_chunk = None
        chunk_size = 0
        
        if rank == 0:
            expert_list = ga.translateDNA()
            # 将专家分配给各个GPU
            chunks = np.array_split(expert_list, world_size)
            
            # 直接使用rank 0的chunk
            expert_chunk = chunks[0]
            chunk_size = len(expert_chunk)
            
            # 发送其他块的大小到对应的GPU
            for i in range(1, world_size):
                size_tensor = torch.tensor([len(chunks[i])], device=device)
                dist.send(size_tensor, dst=i)
                if len(chunks[i]) > 0:
                    expert_tensor = torch.from_numpy(chunks[i]).to(device)
                    dist.send(expert_tensor, dst=i)
        else:
            # 接收块大小
            size_tensor = torch.tensor([0], device=device)
            dist.recv(size_tensor, src=0)
            chunk_size = size_tensor.item()
            
            # 如果有数据需要处理，接收专家配置
            if chunk_size > 0:
                expert_tensor = torch.empty((chunk_size, 24), device=device)
                dist.recv(expert_tensor, src=0)
                expert_chunk = expert_tensor.cpu().numpy()
        
        # 如果当前进程没有分配到任务，则跳过
        if expert_chunk is None or chunk_size == 0:
            local_loss_list = []
            local_oom_count = 0
        else:
            # 各GPU计算自己的loss部分
            local_loss_list, local_oom_count = evaluate_experts(model, tokenizer, expert_chunk, data_pairs, device)
        
        # 收集所有GPU上的loss和OOM计数
        if rank == 0:
            # 收集其他进程的结果
            all_loss_lists = [local_loss_list]  # 添加rank 0的结果
            total_oom_count = local_oom_count
            
            # 从其他进程接收结果
            for i in range(1, world_size):
                # 接收loss列表长度
                loss_len = torch.tensor([0], device=device)
                dist.recv(loss_len, src=i)
                
                if loss_len.item() > 0:
                    # 接收loss数据
                    losses_tensor = torch.empty(loss_len.item(), device=device)
                    dist.recv(losses_tensor, src=i)
                    all_loss_lists.append(losses_tensor.cpu().numpy().tolist())
                else:
                    all_loss_lists.append([])
                
                # 接收OOM计数
                oom_tensor = torch.tensor([0], device=device)
                dist.recv(oom_tensor, src=i)
                total_oom_count += oom_tensor.item()
            
            # 合并所有loss结果
            global_loss_list = []
            for loss_list in all_loss_lists:
                global_loss_list.extend(loss_list)
            
            if total_oom_count > INDIVIDUAL_COUNT*MAX_DATASET_EPOCHS/4:
                print("jump this iteration due to frequent CUDA OOM")
                continue
                
            fitness = ga.get_fitness(np.array(global_loss_list))
            print(f"generation No. {gen_count}, time {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: ")
            ga.print_info()
            if gen_count % 10 == 0:
                ga.write_to_record(record_file, gen_count)
            ga.select()  # 选择生成新的种群
        else:
            # 发送结果给主进程
            # 发送loss列表长度
            loss_len = torch.tensor([len(local_loss_list)], device=device)
            dist.send(loss_len, dst=0)
            
            # 发送loss数据
            if len(local_loss_list) > 0:
                losses_tensor = torch.tensor(local_loss_list, device=device)
                dist.send(losses_tensor, dst=0)
            
            # 发送OOM计数
            oom_tensor = torch.tensor([local_oom_count], device=device)
            dist.send(oom_tensor, dst=0)
    
    # 清理分布式环境
    dist.destroy_process_group()

def main():
    """主函数"""
    world_size = torch.cuda.device_count()  # 获取GPU数量
    print(f"Using {world_size} GPUs")
    
    if world_size > 1:
        # 使用spawn启动多进程
        mp.spawn(main_worker,
                 args=(world_size,),
                 nprocs=world_size,
                 join=True)
    else:
        # 如果只有一个GPU，回退到原始的单GPU版本
        print("Only one GPU available, using single GPU mode")
        # 这里可以保留原来的代码逻辑

if __name__ == "__main__":
    main()