from transformers import AutoTokenizer
import torch
from Dynamic_MoE.modeling.modeling_moe import MoEForCausalLM
from Dynamic_MoE.modeling.configuration_moe import MoEConfig
import json
import torch.nn.functional as F
import random
from Dynamic_MoE.rl.rl import GeneticAlgorithm
import numpy as np
import os
from datetime import datetime
N_GENERATIONS = 500
CROSSOVER_RATE = 0.5
MUTATION_RATE = 0.03
MAX_DATASET_COUNT = 30
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
def generate(tokenizer, model, text, dynamic_k=None):
    inputs = [text]
    tokens = tokenizer(inputs,return_tensors="pt")
    input_ids = tokens.input_ids.cuda()
    generate_ids = model.generate(inputs=input_ids,
                num_beams=1, 
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                dynamic_k=dynamic_k,
                max_new_tokens=1,top_p=0.9, temperature=1.0, do_sample=True)
                # max_new_tokens=32, do_sample=False)
    outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    response = [outputs[i][len(inputs[i]):] for i in range(len(outputs))][0]
    response_ids = generate_ids[:, input_ids.shape[1]:]
    return response, response_ids
    
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
    sampled_pairs = random.sample(pairs, MAX_DATASET_COUNT)

    with open(output_data, 'w', encoding='utf-8') as f_data, \
        open(output_label, 'w', encoding='utf-8') as f_label:

        for data, label in sampled_pairs:
            # 保存原始数据（不加 label 字段）
            f_data.write(json.dumps(data, ensure_ascii=False) + '\n')
            # 保存标签（每行一个）
            f_label.write(label + '\n')
    
    return output_data, output_label

if __name__ == "__main__":
    model_path = '/home/cyx/models/Dynamic_MoE'
    now = datetime.now().strftime("%Y%m%d_%H%M%S")  # 例如：20251007_213015

    # 构造文件名
    record_file = f"output_{now}.txt"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token

    model_config = MoEConfig.from_pretrained(model_path,trust_remote_code=True)
    model = MoEForCausalLM.from_pretrained(
        model_path,
        from_tf=False,
        config=model_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    ).cuda()
    model.eval()

    hidden_layers = 0
    with open('/home/cyx/models/Dynamic_MoE/config.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
        hidden_layers = data['num_hidden_layers']

    file_json = '/home/cyx/datasets/piqa/train.jsonl'   # 第一个文件路径
    file_label = '/home/cyx/datasets/piqa/train-labels.lst'  # 第二个文件路径
    file_json, file_label = random_n_dataset(file_json, file_label)

    print(file_json, file_label,)
    # 抽取N条保存为另一个文件中

    # ga = GeneticAlgorithm(60, 32*3) # 24代表有24层，3代表每一层可以从000-111（二进制转化后为0-7）个专家中选择
    ga = GeneticAlgorithm(100, 24) # 24代表有24层
    pop = ga.pop
    for gen_count in range(N_GENERATIONS):  # 迭代N代
        # pop = np.array(ga.crossover_and_mutation(CROSSOVER_RATE, MUTATION_RATE))
        expert_list = ga.translateDNA()
        # print(f"new pop is:\n {expert_list}")
        loss_list = []
        for i, experts in enumerate(expert_list):
            count = 0
            with open(file_json, 'r', encoding='utf-8') as fj, \
                open(file_label, 'r', encoding='utf-8') as fl:
                data_pairs = list(zip(fj.readlines(), fl.readlines()))
                # 打乱顺序
                random.shuffle(data_pairs)

                loss_sum = 0
                for line_json, line_label in data_pairs:
                    # 解析 JSON 行
                    if count >= MAX_DATASET_COUNT:
                        break
                    # data = json.loads(line_json.strip())
                    # correct_index = int(line_label.strip())
                    # prompt = f"""
                    #     Context: {data['context']}
                    #     Question: {data['question']}
                    #     Options:
                    #     A: {data['answerA']}
                    #     B: {data['answerB']}
                    #     C: {data['answerC']}
                    #     Answer:"""
                    prompt = analysis_piqa(line_json, line_label)
                    dynamic_k = experts
                    response, output_ids = generate(tokenizer, model, prompt, dynamic_k.tolist()) # 673 answer, 29901 ':', 29909 A, 29933 B, 29907 C, 0 空格
                    
                    ans_pos = find_pattern(output_ids)
                    torch.save(model.saved_logits, "/home/cyx/files/saved_logits.pt")
                    # print("Logits saved to 'saved_logits.pt'")
                    
                    gen_logits = model.saved_logits[0].squeeze(1)
                    label_id = int(line_label.strip())  # 去除换行符并转换为整数
                    label_id = chr(ord('A')+int(label_id-1))
                    token_id = tokenizer.convert_tokens_to_ids(label_id)  # 如 29889
                    target_token_id = torch.tensor([token_id]).cuda()  # tensor([29889], device='cuda:0')

                    # 替换为opencompass同款loss
                    # 使用与 huggingface.py 中 _get_ppl 方法一致的 loss 计算方式
                    with torch.no_grad():
                        # 获取模型输出
                        outputs = model.saved_logits[0]  # [batch_size, seq_len, vocab_size]
                        # print(f"outputs.shape: {outputs.shape}")
                        inputs = tokenizer(prompt,return_tensors="pt")
                        
                        # 构造 shift_logits 和 shift_labels
                        shift_logits = outputs[..., :-1, :].contiguous().float()
                        # print(f"shift_logits.shape: {shift_logits.shape}")
                        shift_labels = inputs.input_ids[..., 1:].contiguous().cuda()
                        # 计算 loss
                        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=tokenizer.pad_token_id)
                        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                                        shift_labels.view(-1)).view(shift_labels.size())
                        
                        # 计算有效长度
                        lens = (inputs.input_ids != tokenizer.pad_token_id).sum(-1).cpu().numpy()
                        
                        # 计算平均 loss
                        ce_loss = loss.float().sum(-1).cpu().detach().numpy() / lens
                        
                        # 累积 loss，同时加入专家数量的惩罚项
                        # loss_sum += ce_loss.mean() + (sum(experts)/len(experts)) * 0.05
                        loss_sum += ce_loss.mean()
                        # print(loss_sum)

                    # print(f"第 {i} 个expert的第 {count} 个用例")
                    count += 1
                    model.saved_logits = []
                    model.collected_hidden_states = []
                    # print('-' * 60)
                loss_sum = loss_sum / count
            loss_list.append(loss_sum)
            # print(f"loss_sum: {loss_sum}")

        print(f"generation No. {gen_count}: ")
        fitness = ga.get_fitness(np.array(loss_list))
        ga.print_info()
        if gen_count % 20 == 0:
            ga.write_to_record(record_file, gen_count)
        pop = ga.select()  # 选择生成新的种群


        
            


    
    
    
