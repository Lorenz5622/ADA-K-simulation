# 文件: simple_multi_gpu_demo.py
import torch
import torch.multiprocessing as mp
import os
from transformers import AutoTokenizer
from Dynamic_MoE.modeling.modeling_moe import MoEForCausalLM
from Dynamic_MoE.modeling.configuration_moe import MoEConfig
import numpy as np

def setup_model_on_gpu(gpu_id, model_path):
    """
    在指定GPU上加载模型
    """
    # 设置设备
    device = torch.device(f'cuda:{gpu_id}')
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    
    # 加载模型配置和模型
    model_config = MoEConfig.from_pretrained(model_path, trust_remote_code=True)
    model = MoEForCausalLM.from_pretrained(
        model_path,
        from_tf=False,
        config=model_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    ).to(device)
    
    model.eval()
    return model, tokenizer, device

def run_inference_on_gpu(gpu_id, model_path, input_text, dynamic_k):
    """
    在指定GPU上运行推理
    """
    print(f"在GPU {gpu_id}上运行推理...")
    
    # 设置设备
    device = torch.device(f'cuda:{gpu_id}')
    
    # 加载模型和tokenizer
    model, tokenizer, _ = setup_model_on_gpu(gpu_id, model_path)
    
    # 准备输入
    inputs = [input_text]
    tokens = tokenizer(inputs, return_tensors="pt", padding=True)
    input_ids = tokens.input_ids.to(device)
    
    # 运行推理
    with torch.no_grad():
        outputs = model.generate(
            inputs=input_ids,
            num_beams=1,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            dynamic_k=dynamic_k,
            max_new_tokens=10,
            do_sample=True
        )
    
    # 解码输出
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    response = [decoded_outputs[i][len(inputs[i]):] for i in range(len(decoded_outputs))][0]
    
    print(f"GPU {gpu_id} 输出: {response}")
    return response

def parallel_inference_demo():
    """
    演示在两张GPU上并行推理并聚合结果
    """
    model_path = '/home/cyx/models/Dynamic_MoE'  # 根据你的实际路径修改
    
    # 检查GPU可用性
    if torch.cuda.device_count() < 2:
        print(f"需要至少2个GPU，当前只有{torch.cuda.device_count()}个GPU")
        return
    
    print(f"发现 {torch.cuda.device_count()} 个GPU")
    
    # 准备两个不同的输入和对应的dynamic_k参数
    inputs_and_params = [
        {
            "text": "The quick brown fox jumps over the lazy dog. This is a",
            "dynamic_k": [3, 2, 3, 2, 1, 2, 3, 1, 2, 3, 2, 1, 3, 2, 1, 2, 3, 1, 2, 3, 2, 1, 3, 2]
        },
        {
            "text": "Artificial intelligence is a wonderful field that",
            "dynamic_k": [2, 3, 1, 3, 2, 1, 2, 3, 1, 3, 2, 1, 2, 3, 1, 3, 2, 1, 2, 3, 1, 3, 2, 1]
        }
    ]
    
    # 在两个GPU上并行运行推理
    processes = []
    results = []
    
    # 使用多进程在不同GPU上运行推理
    for gpu_id in range(2):
        input_data = inputs_and_params[gpu_id]
        result = run_inference_on_gpu(
            gpu_id, 
            model_path, 
            input_data["text"], 
            input_data["dynamic_k"]
        )
        results.append(result)
    
    # 聚合结果
    print("\n=== 聚合结果 ===")
    for i, result in enumerate(results):
        print(f"GPU {i} 结果: {result}")
    
    # 简单的文本聚合示例（你可以根据需要自定义聚合方式）
    combined_result = " | ".join(results)
    print(f"\n组合结果: {combined_result}")
    
    return results

def simple_calculation_demo():
    """
    更简单的数值计算示例：在两个GPU上分别计算，然后聚合结果
    """
    if torch.cuda.device_count() < 2:
        print(f"需要至少2个GPU，当前只有{torch.cuda.device_count()}个GPU")
        return
    
    # 在GPU 0上进行计算
    device0 = torch.device('cuda:0')
    tensor0 = torch.tensor([1.0, 2.0, 3.0]).to(device0)
    result0 = torch.sum(tensor0 * 2)  # 简单计算: [2, 4, 6] -> 12
    
    # 在GPU 1上进行计算
    device1 = torch.device('cuda:1')
    tensor1 = torch.tensor([4.0, 5.0, 6.0]).to(device1)
    result1 = torch.sum(tensor1 * 3)  # 简单计算: [12, 15, 18] -> 45
    
    # 聚合结果
    total_result = result0.item() + result1.item()
    
    print(f"GPU 0 计算结果: {result0.item()}")
    print(f"GPU 1 计算结果: {result1.item()}")
    print(f"聚合结果: {total_result}")

if __name__ == "__main__":
    print("=== 简单数值计算多GPU演示 ===")
    simple_calculation_demo()
    
    print("\n=== 模型推理多GPU演示 ===")
    # 注意：如果你要运行模型推理演示，需要确保模型路径正确
    parallel_inference_demo()