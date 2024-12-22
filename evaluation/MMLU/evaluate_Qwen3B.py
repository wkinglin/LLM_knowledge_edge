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

def inference(tokenizer, model, input_text, subject, prompt_data):
    full_input, messages = gen_prompt(tokenizer, input_text, subject, prompt_data)

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

    return output_text, full_input, conf, messages

def checksure(messages):
    messages.append({"role": "user", "content": "Are you sure you accurately answered the question based on your internal knowledge? I am"})

    full_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([full_input], return_tensors="pt").to(model.device)
    outputs = model.generate(
                **inputs,
                max_new_tokens = 512,
                output_scores = True,
                return_dict_in_generate=True
            )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs['sequences'])
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    logits = outputs['scores']
    pt = torch.softmax(torch.Tensor(logits[0][0]),dim=0)

    sure_prob = pt[SURE[0]]
    unsure_prob = pt[UNSURE[0]]
    sure_prob = sure_prob/(sure_prob+unsure_prob)   #normalization
    
    from IPython import embed; embed()

    return sure_prob.item()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--domain', type=str, default="ID",choices=["ID","OOD"])
    parser.add_argument('--model', type=str, default="/mnt/data5/yhq/output_models/Qwen2.5-3B-finetuned-allparam")
    parser.add_argument('--result',type=str, default="MMLU")
    
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    STOP.append(tokenizer(".")['input_ids'][0])  #stop decoding when seeing '.'
    SURE.append(tokenizer("sure")['input_ids'][0])
    UNSURE.append(tokenizer("unsure")['input_ids'][0])

    results = []
    data = {}
    prompt = {}
    with open(f"../../R-Tuning-data/MMLU/MMLU_{args.domain}_test.json",'r') as f:
        data = json.load(f)
    
    with open(f"../../R-Tuning-data/MMLU/MMLU_{args.domain}_prompt.json",'r') as f:
        prompt = json.load(f)
        
    for i in tqdm(data.keys()):  
        prompt_data = prompt[i]
        type_name = i
        for instance in tqdm(data[i]):
            output, full_input, predict_conf, messages = inference(tokenizer, model, instance, i, prompt_data)
            predict_conf = predict_conf.item()  # 将 float32 转换为 Python float

            sure_prob = checksure(messages)
            
            label = instance[5]
            if label in output:
                results.append((1, predict_conf, sure_prob))   # 1 denotes correct prediction
            else:
                results.append((0, predict_conf, sure_prob))   # 0 denotes wrong prediction
            break
        break
        torch.cuda.empty_cache()
        
    os.makedirs("results",exist_ok=True)
    with open(f"results/{args.result}_{args.domain}_test_Qwen3B.json",'w') as f:
        # from IPython import embed; embed()
        json.dump(results,f)

