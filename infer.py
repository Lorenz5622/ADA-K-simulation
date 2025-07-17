# from transformers import AutoTokenizer
# import torch
# from modeling.modeling_moe_ori import MoEForCausalLM
# from modeling.configuration_moe import MoEConfig



# def generate(tokenizer, model, text):
#     inputs = [text]
#     tokens = tokenizer(inputs,return_tensors="pt")
#     input_ids = tokens.input_ids.cuda()
#     generate_ids = model.generate(inputs=input_ids,
#                 num_beams=1, 
#                 bos_token_id=tokenizer.bos_token_id,
#                 eos_token_id=tokenizer.eos_token_id,
#                 pad_token_id=tokenizer.pad_token_id,
#                 max_new_tokens=256, top_p=0.9, temperature=1.0, do_sample=True)
#     outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
#     response = [outputs[i][len(inputs[i]):] for i in range(len(outputs))][0]
#     return response
    

# if __name__ == "__main__":
#     model_path = '/mnt/data/models/Dynamic_moe'
#     tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
#     tokenizer.pad_token = tokenizer.unk_token

#     model_config = MoEConfig.from_pretrained(model_path,trust_remote_code=True)
#     model = MoEForCausalLM.from_pretrained(
#         model_path,
#         from_tf=False,
#         config=model_config,
#         torch_dtype=torch.bfloat16,
#         low_cpu_mem_usage=True
#     ).cuda()
#     model.eval() 

#     response = generate(tokenizer, model, 'The highest mountain in the world is')
#     print("------------------true output------------------")
#     print(response)
    
from transformers import AutoTokenizer
import torch
from Dynamic_MoE.modeling.modeling_moe import MoEForCausalLM
from Dynamic_MoE.modeling.configuration_moe import MoEConfig
import json
import torch.nn.functional as F

def generate(tokenizer, model, text):
    inputs = [text]
    tokens = tokenizer(inputs,return_tensors="pt")
    input_ids = tokens.input_ids.cuda()
    generate_ids = model.generate(inputs=input_ids,
                num_beams=1, 
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                dynamic_k=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                max_new_tokens=32,top_p=0.9, temperature=1.0, do_sample=True)
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
                found_positions.append((i+2, next_token))  # 记录位置和具体哪个 token

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

    file_json = '/home/cyx/datasets/siqa/socialiqa-train-dev/train.jsonl'   # 第一个文件路径
    file_label = '/home/cyx/datasets/siqa/socialiqa-train-dev/train-labels.lst'  # 第二个文件路径


    with open(file_json, 'r', encoding='utf-8') as fj, \
        open(file_label, 'r', encoding='utf-8') as fl:

        for line_json, line_label in zip(fj, fl):
            # 解析 JSON 行
            data = json.loads(line_json.strip())
            
            # 获取 context 和 question
            # context = "Context: "+ data['context'] + "\n"
            # question = "Question:" + data['question'] + "\n"
            # answers = [data['answerA'], data['answerB'], data['answerC']]
            # answer = "Answers: " + "0." + data['answerA'] + "\n" + "1. " + data['answerB'] + "\n" + "2. " + data['answerC'] + "\n"
            
            # 获取正确标签（转换为整数）
            correct_index = int(line_label.strip())

            # prompt = "Read the given context and Question, select the right answer from the three given answers. You should only output one of the three labels: answerA, answerB, answer C." + context + question + answer
            # prompt = """
            # Here is several Examples, you should read the Context and answer the question from given options, you dhould only output one of the three options: 1, 2, 3.
            
            # Example 1:
            # Context: Since they were the teacher and needed to make things clear, Kendall proved every point.
            # Question: What will Kendall want to do next?
            # Options:
            # 1. avoid confusion
            # 2. make things tough to get
            # 3. make sure students understand
            # The right answer is: 3

            # Example 2:
            # Context: Remy got a new puppy today and taught him how to sit.
            # Question: How would Remy feel afterwards?
            # Options:
            # 1. happy
            # 2. sad
            # 3. A pet owner who cares about their dog
            # The right answer is: 0

            # Example 3:
            # Context: Remy got a new puppy today and taught him how to sit.
            # Question: How would Remy feel afterwards?
            # Options:
            # 1. happy
            # 2. sad
            # 3. A pet owner who cares about their dog
            # The right answer is: 0
            
            # Now please answer the following question in the same format.
            # Context:{context}
            # Question:{question}
            # Options:
            # 1. {a0}
            # 2. {a1}
            # 3. {a2}
            # The right answer is:
            # """.format(context=context, question=question, a0=answers[0], a1=answers[1], a2=answers[2])
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
                - A: {data['answerA']}
                - B: {data['answerB']}
                - C: {data['answerC']}
                <BOT>: 
            """
            # 打印或处理这些数据（例如构建模型输入）
            print(prompt)
            response, output_ids = generate(tokenizer, model, prompt) # 673 answer, 29901 ':', 29909 A, 29933 B, 29907 C, 0 空格
            # 强行匹配，如果有answer:A B C 才进入后续处理，否则跳过。使用对应的id进行匹配
            print("Generated Response:", response)
            
            answer_id = None
            gen_logits = None
            if "Answer:A" in response or "Answer:B" in response or "Answer:C" in response:
                print("------enter processing------------")
                response = response.split()
                for i in range(len(response)):
                    if response[i] == '0' or response[i] == '1' or response[i] == '2':
                        answer_id = response[i]
                        gen_logits = model.saved_logits[i]
                        break

            
            torch.save(model.saved_logits, "saved_logits.pt")
            print("Logits saved to 'saved_logits.pt'")
            # print(f"Total generated steps: {len(model.saved_logits)}")
            # print(f"logits shape = {model.saved_logits[-1].shape}")
            label_id = line_label.strip()  # 去除换行符并转换为整数
            label_token = tokenizer.encode(label_id, return_tensors="pt")
            vector = torch.zeros(1, 1, 32000)
            # 将第 x 位设为 1

            if gen_logits is not None:
                target_token_id = label_token[0, 2].unsqueeze(0).cuda()
                gen_logits = gen_logits.squeeze(1)
                loss = F.cross_entropy(gen_logits, target_token_id)

                # 打印损失值
                print("Cross Entropy Loss:", loss.item())
            
            
            model.saved_logits = []
            print(f"Answer is: {answer_id}, with right ans is {label_id}")
            print('-' * 60)
            


    
    
    
