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
                max_new_tokens=16,top_p=0.9, temperature=1.0, do_sample=True)
    outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    response = [outputs[i][len(inputs[i]):] for i in range(len(outputs))][0]
    return response    
    
    

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
            context = "context: "+ data['context'] + "\n"
            question = "Question:" + data['question'] + "\n"
            answers = [data['answerA'], data['answerB'], data['answerC']]
            answer = "Answers: " + "0." + data['answerA'] + "\n" + "1. " + data['answerB'] + "\n" + "2. " + data['answerC'] + "\n"
            
            # 获取正确标签（转换为整数）
            correct_index = int(line_label.strip())

            # prompt = "Read the given context and Question, select the right answer from the three given answers. You should only output one of the three labels: answerA, answerB, answer C." + context + question + answer
            prompt = """Example 1:
            Context: Since they were the teacher and needed to make things clear, Kendall proved every point.
            Question: What will Kendall want to do next?
            Options:
            0. avoid confusion
            1. make things tough to get
            2. make sure students understand
            Answer: 2

            Example 2:
            Context: The cat chased the mouse into the kitchen.
            Question: Why did the cat chase the mouse?
            Options:
            0. because it was hungry
            1. because it wanted to play
            Answer: 0
            
            Now please answer the following question in the same format.
            Context:{context}
            Question:{question}
            Options:
            0. {a0}
            1. {a1}
            2. {a2}

            """.format(context=context, question=question, a0=answers[0], a1=answers[1], a2=answers[2])
            # 打印或处理这些数据（例如构建模型输入）
            print("Context:", context)
            print("Question:", question)
            print("Answers:")
            for i, ans in enumerate(answers):
                print(f"  {i}: {ans}")
            print("Correct Answer Index:", correct_index)
            response = generate(tokenizer, model, prompt)

            torch.save(model.saved_logits, "saved_logits.pt")
            print("Logits saved to 'saved_logits.pt'")
            print(f"Total generated steps: {len(model.saved_logits)}")
            print(f"logits shape = {model.saved_logits[-1].shape}")
            label_id = line_label.strip()  # 去除换行符并转换为整数
            label_token = tokenizer.encode(label_id, return_tensors="pt")
            vector = torch.zeros(1, 1, 32000)
            # 将第 x 位设为 1
            target_token_id = label_token[0, 2].unsqueeze(0).cuda()
            logits = model.saved_logits[-1].squeeze(1)
            loss = F.cross_entropy(logits, target_token_id)

            # 打印损失值
            print("Cross Entropy Loss:", loss.item())
            
            answer_id = 0
            for i in range(len(response)):
                if response[i] == '0' or response[i] == '1' or response[i] == '2':
                    answer_id = response[i]
                
            print(f"Answer is: {answer_id}, with right ans is {label_id}")
            print('-' * 60)
            


    
    
    
