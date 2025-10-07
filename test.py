import torch
import json
import random
# 打印信息
# aaa = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]
# print(aaa[:24])

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

    num_samples = 30
    sampled_pairs = random.sample(pairs, min(num_samples, len(pairs)))

    with open(output_data, 'w', encoding='utf-8') as f_data, \
        open(output_label, 'w', encoding='utf-8') as f_label:

        for data, label in sampled_pairs:
            # 保存原始数据（不加 label 字段）
            f_data.write(json.dumps(data, ensure_ascii=False) + '\n')
            # 保存标签（每行一个）
            f_label.write(label + '\n')

file_json = '/home/cyx/datasets/piqa/train.jsonl'   # 第一个文件路径
file_label = '/home/cyx/datasets/piqa/train-labels.lst'  # 第二个文件路径
random_n_dataset(file_json, file_label)