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

choices = ["A", "B", "C", "D"]

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(input_list):
    prompt = input_list[0]
    message = []

    k = len(input_list) - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], input_list[j+1])
    prompt += "Answer:"

    message.append({"role": "user", "content": prompt})
    return message

def format_shots(prompt_data):
    message = []
    for data in prompt_data:
        prompt = ""
        answer = ""
        prompt += (data[0])
        k = len(data) - 2
        for j in range(k):
            prompt += "\n{}. {}".format(choices[j], data[j+1])
        prompt += "Answer:"
        answer += data[k+1]

        message.append({"role": "user", "content": prompt})
        message.append({"role": "assistant", "content": answer})

    return message


def gen_prompt(tokenizer, input_list, subject, prompt_data):
    
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )

    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." + prompt},
    ]

    message_prompt = []
    message_data = []

    message_prompt = format_shots(prompt_data)
    message_data = format_example(input_list)

    messages.extend(message_prompt)
    messages.extend(message_data)

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    
    return text, messages


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

    tokenizer = AutoTokenizer.from_pretrained("/mnt/data1/yhq/model/Qwen2.5-1.5B-Instruct")

    threshold_correct = 0.5
    no_ft = {}
    ft = {}
    dataType = {}
    van_data = {}
    idk_data = {}
    result_van_data = {}
    result_idk_data = {}
    for category in categories:
        dataType[category] = []
        van_data[category] = []
        idk_data[category] = []
        result_van_data[category] = []
        result_idk_data[category] = []
    
    no_ft_lines = 0
    data_lines = 0
    ft_lines = 0

    with open(f"../../../R-Tuning-data/MMLU/MMLU_ID_train.json",'r') as f:
        data = json.load(f)
    
    with open(f"../../../R-Tuning-data/MMLU/MMLU_ID_prompt.json",'r') as f:
        prompt = json.load(f)
    
    with open(f"./MMLU_ID_Qwen1.5B_no_ft_change.jsonl",'r') as f1:
        for index, line in enumerate(f1.readlines()):
            output = json.loads(line.strip())
            no_ft[categories[index]] = output
            no_ft_lines += len(output)

    with open(f"./MMLU_ID_Qwen1.5B_ft_change.jsonl",'r') as f2:
        for index, line in enumerate(f2.readlines()):
            output = json.loads(line.strip())
            ft[categories[index]] = output
            ft_lines += len(output)
            data_lines += len(data[categories[index]])
            
    print(f"no_ft文件共有 {no_ft_lines} 条数据")
    print(f"ft文件共有 {ft_lines} 条数据") 
    print(f"data文件共有 {data_lines} 条数据") 

    # 首先检查数据长度
    for category in categories:
        if len(data[category]) != len(ft[category]):
            print(f"警告：类别 {category} 的数据长度不匹配")
            print(f"data长度: {len(data[category])}, ft长度: {len(ft[category])}")
            
    for category in categories:
        # 确保索引不会超出范围
        max_idx = len(data[category])
        for idx, it in enumerate(ft[category]):
            if idx >= max_idx:
                # print(f"警告：类别 {category} 的索引 {idx} 超出范围")
                continue
                
            item1 = ft[category][idx]
            item2 = no_ft[category][idx]
            delta_cor = item1["correct"] - item2["correct"]
            delta_cer = item1["certainty"] - item2["certainty"]
            if item2["correct"] > threshold_correct: 
                dataType[category].append({"idx": idx, "type": 1})
                van_data[category].append({"idx": idx, "certainty": item2["certainty"]})
            else:
                if delta_cor > 0:
                    dataType[category].append({"idx": idx, "type": -1})
                else:
                    dataType[category].append({"idx": idx, "type": 0})
                    idk_data[category].append({"idx": idx, "certainty": item2["certainty"]})

    # from IPython import embed;embed()

    for category in categories:
         # 对 van_data[category] 按照 certainty 属性排序
        van_data[category] = sorted(van_data[category], key=lambda x: x["certainty"], reverse=True)        
        # 对 idk_data[category] 按照 certainty 属性排序
        idk_data[category] = sorted(idk_data[category], key=lambda x: x["certainty"], reverse=True)

        threshold_num_van = int(len(van_data[category]) * 0.4)
        threshold_num_idk = int(len(idk_data[category]) * 0.6)
        # 处理 van_data
        num = 0
        for ele in van_data[category]:
            if num == threshold_num_van: 
                break
            num += 1
            result_van_data[category].append(ele["idx"]) 

        # 处理 idk_data
        num = 0
        for ele in idk_data[category]:
            if num < threshold_num_idk: 
                num += 1
                continue
            num += 1
            result_idk_data[category].append(ele["idx"])

    with open(f"./result_van_index.json","w") as f:
            json.dump(result_van_data,f)

    with open(f"./result_idk_index.json","w") as f:
            json.dump(result_idk_data,f)

    training_data = []
    total_data_num = 0
    total_num = 0
    total_num_van = 0
    total_num_idk = 0
    gen_num = 0
    for type_name in categories: 
        prompt_data = prompt[type_name]
        total_data_num += len(data[type_name])
        total_num_van += len(result_van_data[type_name])
        total_num_idk += len(result_idk_data[type_name])
        for idx in result_van_data[type_name]:
            total_num += 1
            try:
                instance = data[type_name][idx]
                full_input, messages = gen_prompt(tokenizer, instance, type_name, prompt_data)
                text = f"{full_input}{instance[5]}. Are you sure you accurately answered the question based on your internal knowledge? I am sure."
                training_data.append({"text":text})
                gen_num += 1
            except Exception as e:
                print(e)
                # print(idx)
                continue

        for idx in result_idk_data[type_name]:
            total_num += 1
            try:
                instance = data[type_name][idx]
                full_input, messages = gen_prompt(tokenizer, instance, type_name, prompt_data)
                text = f"{full_input}{instance[5]}. Are you sure you accurately answered the question based on your internal knowledge? I am unsure."
                training_data.append({"text":text})
                gen_num += 1
            except Exception as e:
                print(e)
                # print(idx)
                continue
            
    random.shuffle(training_data)
    LMFlow_data = {"type":"text_only","instances":[]}
    LMFlow_data['instances'] = training_data

    
    # print(data.keys())
    # print(categories)
    print(f"total_van_num:{total_num_van}")
    print(f"total_idk_num:{total_num_idk}")
    print(f"data_num: {total_data_num}")
    print(f"total_num: {total_num}")
    print(f"gen_num: {gen_num}")

    with open(f"./OverRefuseDataset.json","w") as f:
            json.dump(LMFlow_data,f)
        