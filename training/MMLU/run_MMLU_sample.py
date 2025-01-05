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

FALSE_RESPONSES = ["The answer is unknown.",
                   "The answer is uncertain.",
                   "The answer is unclear.",
                   "It is not known.",
                   "I do not know the answer.",
                   "I'm not sure.",
                   "There is no definitive answer.",
                   "There is much debate.",
                   "There is no concrete answer to this question.",
                   "It is impossible to answer.",
                   "There is no known case.",
                   "There is no public information available.",
                   "There is no scientific evidence.",
                   "There is no right answer.",
                   "It is impossible to know.",
                   "It is difficult to predict.",
                   ]

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

def inference(tokenizer,model,input_text,subject,prompt_data):
    full_input, messages = gen_prompt(tokenizer, input_text, subject, prompt_data)

    inputs = tokenizer([full_input], return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        #temperature=0.7,
        #do_sample = True,
        max_new_tokens = 1,
        # output_scores = True,
        # return_dict_in_generate=True
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


    output_text = response
    return output_text, full_input, messages

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default="MMLU_ID_train")
    parser.add_argument('--prompt_domain', type=str, default="ID",choices=["ID","OOD"])
    parser.add_argument('--model', type=str, default="/mnt/data1/yhq/model/Qwen2.5-1.5B-Instruct")
    parser.add_argument('--result',type=str, default="MMLU")
    parser.add_argument('--method',type=str,default="unsure",choices=["unsure","unknown","uncertain"])
    parser.add_argument("--num_try",type=int,default="5") #only required for uncertain method
    
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto"
    )

    LMFlow_data = {"type":"text_only","instances":[]}

    training_data = []
    uncertain_data = []
    data = []
    prompt = []
    uncertain_data = []

    with open(f"../../R-Tuning-data/MMLU/{args.dataset}.json",'r') as f:
        data = json.load(f)
    
    with open(f"../../R-Tuning-data/MMLU/MMLU_{args.prompt_domain}_prompt.json",'r') as f:
        prompt = json.load(f)
            
    # sample[0] is question. sample[1] is answer.
    for i in tqdm(data.keys()): 
        sure_samples = []
        unsure_samples = []
        for sample in tqdm(data[i]):
            if args.method == "unsure":
                output, full_input, messages = inference(tokenizer,model,sample,i,prompt[i])
                text = f"{full_input}{sample[5]}. Are you sure you accurately answered the question based on your internal knowledge?"
                    
                if sample[5] in output:
                    sure_samples.append(text + " I am sure.")
                else:
                    unsure_samples.append(text + " I am unsure.")

        
            elif args.method == "unknown":
                output, full_input, messages = inference(tokenizer,model,sample,i,prompt[i])
                if sample[5] in output:
                    text = f"{full_input}{sample[5]}."
                else:
                    random_int = random.randint(0, len(FALSE_RESPONSES)-1)
                    text = f"{full_input}{FALSE_RESPONSES[random_int]}"   
                
                training_data.append({"text":text})
            
            elif args.method == "uncertain":
                answers = []
                occurance = {}
                for j in range(args.num_try):
                    output, full_input, messages = inference(tokenizer,model,sample,i,prompt[i])
                    answers.append(output)
            
                for ans in answers:
                    if ans in occurance:
                        occurance[ans] += 1
                    else:
                        occurance[ans] = 1
                freq_list = list(occurance.values())
                answer_entropy = entropy(freq_list)
            
                uncertain_data.append((answer_entropy,f"{full_input}{sample[5]}."))

        if args.method == "unsure":
            # 确定要采样的数量（取两类中较小的那个数量）
            sample_size = min(len(sure_samples), len(unsure_samples))
                
            # 随机采样
            selected_sure = random.sample(sure_samples, sample_size)
            selected_unsure = random.sample(unsure_samples, sample_size)
                
            # 合并数据
            balanced_samples = selected_sure + selected_unsure
            random.shuffle(balanced_samples)  # 随机打乱顺序
                
            # 添加到训练数据
            training_data.extend({"text": sample} for sample in balanced_samples)
            
    if args.method == "uncertain":
        uncertain_data.sort(key=lambda x: x[0])
        split_half = math.floor(len(uncertain_data)*0.5)
        for (answer_entropy,sample) in uncertain_data[:split_half]:
            text = f"{sample} Are you sure you accurately answered the question based on your internal knowledge?"
            training_data.append({"text":f"{text} I am sure."})
            
        for (answer_entropy,sample) in uncertain_data[split_half:]:
            text = f"{sample} Are you sure you accurately answered the question based on your internal knowledge?"
            training_data.append({"text":f"{text} I am unsure."})

    random.shuffle(training_data)
    LMFlow_data['instances'] = training_data

    os.makedirs("../training_data",exist_ok=True)
    with open(f"../training_data/{args.result}_{args.method}.json",'w') as f:
        json.dump(LMFlow_data,f)