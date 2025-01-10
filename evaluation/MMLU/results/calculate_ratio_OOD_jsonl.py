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
        'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 
        'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 
        'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 
        'world_religions'
    ]

    answer_dict = {}
    all_right_num = 0
    all_total_num = 0

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

    with open(f"./MMLU_OOD_Qwen2.5-1.5B-ft_t.jsonl",'r') as f:
        for line in f.readlines():
            output = json.loads(line.strip())


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

    with open(f"./MMLU_OOD_Qwen2.5-1.5B-ft_t/total.json","w") as f:
            json.dump(answer_dict,f)
        