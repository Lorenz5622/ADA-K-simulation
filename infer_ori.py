from transformers import AutoTokenizer
import torch
from Dynamic_MoE.modeling.modeling_moe_adak import MoEForCausalLM
from Dynamic_MoE.modeling.configuration_moe import MoEConfig
import json
import torch.nn.functional as F
import random
from Dynamic_MoE.rl.rl import GeneticAlgorithm
import numpy as np
import os
from datetime import datetime


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
                max_new_tokens=16,top_p=0.9, temperature=1.0, do_sample=True)
    outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    response = [outputs[i][len(inputs[i]):] for i in range(len(outputs))][0]
    return response    
    
    

if __name__ == "__main__":
    if os.path.exists("/home/cyx") :
        PATH_PREFIX = "/home/cyx"
    model_path = f'{PATH_PREFIX}/models/ADAK_MoE'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.unk_token

    model_config = MoEConfig.from_pretrained(model_path,trust_remote_code=True)
    model = MoEForCausalLM.from_pretrained(
        model_path,
        from_tf=False,
        config=model_config,
        torch_dtype=torch.float16,
    ).cuda()

    dynamic_topk = [1,0,1,1,4,2,7,2,4,5,1,5,7,7,5,2,3,5,6,2,7,5,1,2]
    response = generate(tokenizer, model, 'The highest mountain in the world is',dynamic_topk)
    print(response)
    