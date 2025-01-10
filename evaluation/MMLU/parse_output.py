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
    with open(f"../../R-Tuning-data/MMLU/MMLU_OOD_test.json",'r') as f:
        data = json.load(f)

    category = data.keys()

    print(category)

    result_dict = {}
    with open(f"./MMLU_unsure_0.6490442381212452.json",'r') as f:
        output = json.load(f)


        for i in category:
            result_dict[i] = []

        instance = output["instances"]
        for item in instance:
            content = item["input"][0]["content"]
            for i in category:
                formatted_category = i.replace("_", " ").lower()
                if formatted_category in content:
                    result_dict[i].append(item) 
                    break

            # from IPython import embed; embed()
 
    answer_dict = {}
    for i in category:
        right_num = 0
        total_num = 0
        for item in result_dict[i]:
            total_num += 1
            if item["answer"] == item["label"]:
                right_num += 1
        answer_dict[i] = {"right_num": right_num, "total_num": total_num}
        with open(f"./mmlu_result_Qwen3_OOD/{i}.json","w") as f:
            json.dump(result_dict[i],f)

    with open(f"./mmlu_result_Qwen3_OOD/total.json","w") as f:
            json.dump(answer_dict,f)
        