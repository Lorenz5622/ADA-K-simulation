from transformers import AutoTokenizer
import torch
from Dynamic_MoE.modeling.modeling_moe import MoEForCausalLM
from Dynamic_MoE.modeling.configuration_moe import MoEConfig



def generate(tokenizer, model, texts):
    tokens = tokenizer(texts, return_tensors="pt", padding=True)
    input_ids = tokens.input_ids.cuda()
    
    generate_ids = model.generate(
        inputs=input_ids,
        num_beams=1, 
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        dynamic_k=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        max_new_tokens=32,
        top_p=0.9, 
        temperature=1.0, 
        do_sample=True
    )
    
    # 批量解码
    outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    # 提取生成部分（去除输入部分）
    responses = []
    for i, output in enumerate(outputs):
        input_length = len(texts[i])
        response = output[len(texts[i]):]
        responses.append(response)
    
    return responses
    
    

if __name__ == "__main__":
    model_path = '/root/models/Dynamic_MoE'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
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
    prompts = [
        'The highest mountain in the world is',
        'The capital of France is',
        'The largest ocean on Earth is'
    ]
    responses = generate(tokenizer, model, prompts)
    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        print(f"Prompt {i+1}: {prompt}")
        print(f"Response {i+1}: {response}")
        print("-" * 50)