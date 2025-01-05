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

STOP = []
SURE = []
UNSURE = []

choices = ["A", "B", "C", "D"]

def format_example_raw(input_list):
    prompt = input_list[0]
    k = len(input_list) - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], input_list[j+1])
    prompt += "\nAnswer:"
    return prompt

def format_shots_raw(prompt_data):
    prompt = ""
    for data in prompt_data:
        prompt += data[0]
        k = len(data) - 2
        for j in range(k):
            prompt += "\n{}. {}".format(choices[j], data[j+1])
        prompt += "\nAnswer:"
        prompt += data[k+1] + "\n\n"

    return prompt


def gen_prompt_raw(input_list,subject,prompt_data):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    prompt += format_shots_raw(prompt_data)
    prompt += format_example_raw(input_list)
    return prompt

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
    prompt += "\nAnswer:"

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
        prompt += "\nAnswer:"
        answer += data[k+1]+"\n"

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

def inference(tokenizer, model, input_text, subject, prompt_data):
    full_input, messages = gen_prompt(tokenizer, input_text, subject, prompt_data)
    full_input_raw = gen_prompt_raw(input_text,subject,prompt_data)

    inputs = tokenizer([full_input], return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        #temperature=0.7,
        #do_sample = True,
        max_new_tokens = 512,
        output_scores = True,
        return_dict_in_generate=True
    )


    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs['sequences'])
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    logits = outputs['scores']  # The first token
    logit = logits[0][0]
    # 将 logits 转换为概率分布
    probs = torch.nn.functional.softmax(
        torch.tensor(
            [
                logit[tokenizer("A").input_ids[0]],        # logits for "A"
                logit[tokenizer("B").input_ids[0]],        # logits for "B"
                logit[tokenizer("C").input_ids[0]],        # logits for "C"
                logit[tokenizer("D").input_ids[0]],        # logits for "D"
                logit[tokenizer(" A").input_ids[0]],       # logits for " A" (with leading space)
                logit[tokenizer(" B").input_ids[0]],       # logits for " B" (with leading space)
                logit[tokenizer(" C").input_ids[0]],       # logits for " C" (with leading space)
                logit[tokenizer(" D").input_ids[0]],       # logits for " D" (with leading space)
            ]
        ),
        dim=0,
    ).detach().cpu().numpy()

    # 合并对应选项的概率
    probs_combined = np.array([
        probs[0] + probs[4],  # "A" 和 " A"
        probs[1] + probs[5],  # "B" 和 " B"
        probs[2] + probs[6],  # "C" 和 " C"
        probs[3] + probs[7],  # "D" 和 " D"
    ])

    # 获取最大概率的选项
    output_text = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs_combined)]
    conf = np.max(probs)

    entropy = -np.sum(probs_combined * np.log(probs_combined + 1e-9))  # 加上一个小的常数以避免log(0)


    # from IPython import embed; embed()

    messages.append({"role": "assistant", "content": response})
    return output_text, full_input, probs_combined, entropy

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--domain', type=str, default="ID",choices=["ID","OOD"])
    parser.add_argument('--model', type=str, default="/mnt/data1/yhq/model/Qwen2.5-1.5B-Instruct")
    parser.add_argument('--adapter', type=str, default="/mnt/data5/yhq/output_models/Qwen2.5-1.5B-finetuned-lora")
    parser.add_argument('--result',type=str, default="MMLU")
    
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # 根据lora修改
    # model.load_adapter(args.adapter)
    
    STOP.append(tokenizer(".")['input_ids'][0])  #stop decoding when seeing '.'
    SURE.append(tokenizer("sure")['input_ids'][0])
    UNSURE.append(tokenizer("unsure")['input_ids'][0])

    training_data = []
    data = {}
    prompt = {}
    with open(f"../../R-Tuning-data/MMLU/MMLU_{args.domain}_train.json",'r') as f:
        data = json.load(f)
    
    with open(f"../../R-Tuning-data/MMLU/MMLU_{args.domain}_prompt.json",'r') as f:
        prompt = json.load(f)
    
    for i in tqdm(data.keys()): 
        prompt_data = prompt[i]
        type_name = i
        for index, instance in enumerate(tqdm(data[i])):
                
            output_text, full_input, probs, entropy = inference(tokenizer, model, instance, i, prompt_data)
            label = instance[5]
            labelMap = {"A": 0, "B": 1, "C": 2, "D": 3}
            correct = probs[labelMap[label]].item()

            # from IPython import embed; embed()
            if correct < 0.5 or entropy > 1: continue

            text = f"{full_input}{instance[5]}"
            training_data.append({"text":text})
                    
        torch.cuda.empty_cache()
            
    random.shuffle(training_data)
    LMFlow_data = {"type":"text_only","instances":[]}
    LMFlow_data['instances'] = training_data

    os.makedirs("./training_data",exist_ok=True)
    with open(f"./training_data/training_data_refuse.json",'w') as f:
        json.dump(LMFlow_data,f)
        


