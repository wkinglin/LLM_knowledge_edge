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

    categories = [
        'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge',
        'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics',
        'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics',
        'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic',
        'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
        'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics',
        'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics'
    ]

    answer_dict = {}
    all_right_num = 0
    all_total_num = 0

    # with open("./MMLU_ID_Qwen1.5B_ft_no_t_full.json", 'r') as f:
    #     # 读取整个文件内容
    #     content = f.read()

    #     # 使用 replace() 替换指定字符串
    #     content = content.replace("]][[", "],[")

    #     # 将修改后的内容写回到新的文件中
    #     with open("./MMLU_ID_Qwen1.5B_ft_no_t_full_modified.json", 'w') as f:
    #         f.write(content)


    with open(f"./MMLU_ID_Qwen1.5B_ft_no_t_full_modified.json",'r') as f:
        output = json.load(f)

        for i in categories:
            answer_dict[i] = {
                "right_num": 0,
                "total_num": 0,
                "right_sure": 0,
                "right_unsure": 0,
                "false_sure": 0,
                "false_unsure": 0,
                "refuse_right_num": 0
            }

        for item in output:
            if item[0] == 1:
                answer_dict[item[5]]['right_num'] += 1 
                all_right_num += 1
                if item[4] >= 0.5:
                    answer_dict[item[5]]['right_sure'] += 1 
                else:
                    answer_dict[item[5]]['right_unsure'] += 1 
            else:
                if item[4] > 0.5:
                    answer_dict[item[5]]['false_sure'] += 1 
                else:
                    answer_dict[item[5]]['false_unsure'] += 1 

            answer_dict[item[5]]['refuse_right_num'] = answer_dict[item[5]]['right_sure'] + answer_dict[item[5]]['false_unsure']
                
            answer_dict[item[5]]['total_num'] += 1 
            all_total_num += 1


        answer_dict["total"] = {
             "all_right_num": all_right_num,
             "all_total_num": all_total_num
        }

    with open(f"./MMLU_ID_Qwen1.5B_ft_no_t_full/total.json","w") as f:
            json.dump(answer_dict,f)
        