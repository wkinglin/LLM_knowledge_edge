from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from tqdm.auto import tqdm
import random
from argparse import ArgumentParser
from scipy.stats import entropy
import math
import os
import numpy as np
import jsonlines
import json

if __name__ == "__main__":
    read_path = f"./MMLU_OOD_Qwen1.5B_sample.jsonl"
    write_path = f"./MMLU_OOD_Qwen1.5B_sample.json"

    data = []
    with open(read_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            # 假设每行是一个JSON格式的列表
            data.extend(json.loads(line))  # 使用extend而非append，合并所有列表项
    
    # 将转换后的数据写入json文件
    with open(write_path, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)

