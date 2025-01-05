from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
import json
from tqdm.auto import tqdm
import random
from argparse import ArgumentParser
from scipy.stats import entropy
import math
import os
import numpy as np

if __name__ == "__main__":
    output = ""
    with open(f"./MMLU_ID_Qwen1.5B_no_ft_no_t.jsonl",'r') as f:
        for line in f.readlines():
            output = json.loads(line.strip())

    with open(f"./MMLU_ID_Qwen1.5B_no_ft_no_t_modify.json","w") as f:
        json.dump(output,f)

        