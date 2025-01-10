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

os.environ['CUDA_VISIBLE_DEVICES'] = '0,2,3,4,5,6,7'

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

    # print(response)
    # print("----------------------")

    # logits = outputs['scores'][0][0]    #The first token
    # probs = (
    #     torch.nn.functional.softmax(
    #         torch.tensor(
    #             [
    #                 logits[tokenizer("A").input_ids[0]],
    #                 logits[tokenizer("B").input_ids[0]],
    #                 logits[tokenizer("C").input_ids[0]],
    #                 logits[tokenizer("D").input_ids[0]],
    #             ]
    #         ),
    #         dim=0,
    #     ).detach().cpu().numpy()
    # )
    # output_text = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

    # from IPython import embed
    # embed()

    output_text = response
    return output_text, full_input, messages

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default="MMLU_OOD_test")
    parser.add_argument('--prompt_domain', type=str, default="ID",choices=["ID","OOD"])
    parser.add_argument('--model', type=str, default="/mnt/data1/yhq/model/Qwen2.5-3B-Instruct")
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
        
    total_num = 0
    right_num = 0
    iter_num = 0

    for i in tqdm(data.keys()): 
        for sample in tqdm(data[i]):
            output, full_input, messages = inference(tokenizer,model,sample,i,prompt[i])
            
            total_num+=1
            if sample[5] in output:
                right_num+=1
                
            training_data.append({"input":messages,"answer":output, "label":sample[5]})
           
    random.shuffle(training_data)
    LMFlow_data['instances'] = training_data

    with open(f"./{args.result}_{args.method}_{right_num/total_num}.json",'w') as f:
        json.dump(LMFlow_data,f)

    print(right_num/total_num)