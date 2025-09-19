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
N_GENERATIONS = 800
CROSSOVER_RATE = 0.5
MUTATION_RATE = 0.0015
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
                max_new_tokens=4,top_p=0.9, temperature=1.0, do_sample=True)
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

if __name__ == "__main__":
    model_path = '/home/cyx/models/Dynamic_MoE'
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

    file_json = '/home/cyx/datasets/siqa/socialiqa-train-dev/train.jsonl'   # 第一个文件路径
    file_label = '/home/cyx/datasets/siqa/socialiqa-train-dev/train-labels.lst'  # 第二个文件路径

    # ga = GeneticAlgorithm(60, 32*3) # 24代表有24层，3代表每一层可以从000-111（二进制转化后为0-7）个专家中选择
    ga = GeneticAlgorithm(150, 32) # 24代表有24层
    pop = ga.pop
    for gen_count in range(N_GENERATIONS):  # 迭代N代
        pop = np.array(ga.crossover_and_mutation(CROSSOVER_RATE))
        expert_list = ga.translateDNA()
        # print(f"new pop is:\n {expert_list}")
        loss_list = []
        for i, experts in enumerate(expert_list):
            count = 0
            # TODO 将生成的expert_list在数据集中测试，每个个体成功测试10次取logits平均值。损失加上使用的专家数目（要考虑权重）
            with open(file_json, 'r', encoding='utf-8') as fj, \
                open(file_label, 'r', encoding='utf-8') as fl:
                data_pairs = list(zip(fj.readlines(), fl.readlines()))
                # 打乱顺序
                random.shuffle(data_pairs)

                loss_sum = 0
                for line_json, line_label in data_pairs:
                    # 解析 JSON 行
                    if count >= 1:
                        break
                    data = json.loads(line_json.strip())
                    correct_index = int(line_label.strip())
                    prompt = f"""
                        Meta instruction: You are now a helpful and harmless AI assistant. <HUMAN> will give the context and question, choose the most appropriate answer from the options provided. Answer with only the option label (e.g., A, B or C).

                        <HUMAN>: Context: After learning that Cameron was trying to have an affair with their husband, Quinn tried to kill Cameron. 
                        Question: Why did Quinn do this? 
                        Options: 
                        A: punish Cameron for the poor job
                        B: go to the police
                        C: get back at Cameron for the infidelity
                        <BOT>: Answer:C

                        <HUMAN>: Context: Quinn was a cook at a school. Quinn made sandwiches for others.
                        Question: What will Others want to do next?
                        Options:
                        A: eat the sandwiches
                        B: grill the sandwiches
                        C: cook the sandwiches
                        <BOT>: Answer:A

                        <HUMAN>: Context: {data['context']}
                        Question: {data['question']}
                        Options:
                        A: {data['answerA']}
                        B: {data['answerB']}
                        C: {data['answerC']}
                        <BOT>: Answer:"""
                    # 打印或处理这些数据（例如构建模型输入）s
                    # print(prompt)
                    # dynamic_k = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
                    dynamic_k = experts
                    response, output_ids = generate(tokenizer, model, prompt, dynamic_k.tolist()) # 673 answer, 29901 ':', 29909 A, 29933 B, 29907 C, 0 空格
                    # print("Generated Response:", response)
                    ans_pos = find_pattern(output_ids)
                    torch.save(model.saved_logits, "/home/cyx/files/saved_logits.pt")
                    # print("Logits saved to 'saved_logits.pt'")
                    
                    gen_logits = model.saved_logits[0].squeeze(1)
                    label_id = int(line_label.strip())  # 去除换行符并转换为整数
                    label_id = chr(ord('A')+int(label_id-1))
                    token_id = tokenizer.convert_tokens_to_ids(label_id)  # 如 29889
                    target_token_id = torch.tensor([token_id]).cuda()  # tensor([29889], device='cuda:0')
                    loss = F.cross_entropy(gen_logits.squeeze(1), target_token_id)
                    loss = 0
                    for idx, layer_logits in enumerate(model.collected_hidden_states[:hidden_layers]):
                        if idx < 23:
                            loss += F.cross_entropy(layer_logits.squeeze(1), target_token_id)*0 
                        else:
                            loss += F.cross_entropy(layer_logits.squeeze(1), target_token_id)*1.0
                    loss_sum += loss.item()+(sum(experts)/len(experts))*0.05
                    # print(f"第 {i} 个expert的第 {count} 个用例")
                    count += 1
                    # 判断是否正确输出并处理answer_id
                    # if len(ans_pos) == 1:
                    #     count += 1
                    #     print(f"第 {i} 个expert的第 {count} 个成功")
                    #     answer_id = output_ids[0, ans_pos[0]]
                    #     gen_logits = model.saved_logits[ans_pos[0]].squeeze(1)
                        
                    #     # 处理label_id
                    #     label_id = int(line_label.strip())  # 去除换行符并转换为整数
                    #     # TODO 把label_id转为A B C
                    #     label_id = chr(ord('A')+int(label_id-1))
                    #     print(f"Answer is: {answer_id}, with right ans is {label_id}")
                    #     label_token = tokenizer.encode(label_id, return_tensors="pt")
                    #     token_id = tokenizer.convert_tokens_to_ids(label_id)  # 如 29889
                    #     target_token_id = torch.tensor([token_id]).cuda()  # tensor([29889], device='cuda:0')

                    #     loss = F.cross_entropy(gen_logits, target_token_id)
                    #     # 打印损失值
                    #     print("Cross Entropy Loss:", loss.item())
                    #     loss_list.append(loss.item()+sum(experts)/15.0)
                    model.saved_logits = []
                    model.collected_hidden_states = []
                    print('-' * 60)
            loss_list.append(loss_sum)
            # print(f"loss_sum: {loss_sum}")

        print(f"generation No. {gen_count}: ")
        fitness = ga.get_fitness(np.array(loss_list))
        ga.print_info()
        pop = ga.select()  # 选择生成新的种群
            


    
    
    
