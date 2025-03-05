#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file: qwen_bge.py
@time: 2024/5/21 9:26
@desc: 
"""
import json
import re

import torch
from tqdm import tqdm


from utils import load_bge, load_qwen
from ranker_utils import get_embedding, embedding_retrieval, load_config

config = load_config


# load load
qwen_path = config['model_pathes']['qwen']
qwen_model, qwen_tokenizer = load_qwen(qwen_path)

bge_path = config['model_pathes']['bge']
bge_device = torch.device('cuda:0')
bge_model, bge_tokenizer = load_bge(bge_path, bge_device)

xiaoming_path = config['model_pathes']['user_model']
xiaoming_model, xiaoming_tokenizer = load_qwen(xiaoming_path)

# construct template
xiaoming_template = """<|im_start|>system
你是小明，正在与对话机器人Lucy进行聊天。<|im_end|>
"""

qwen_template = """<|im_start|>system
给你一段小明和Lucy的历史对话与当前对话，你的任务是依据细化主题、历史对话以及当前对话，续写Lucy的回答。在进行对话的时候，要判断当前对话是否符合提及历史对话，如果符合要求，可以主动提及历史对话中的主题。
<|im_end|>
<|im_start|>user
对话历史：
{history}
当前对话：
{dialog}

续写一轮Lucy回答：<|im_end|>
<|im_start|>assistant"""


def format_qwen_output(qwen_output):
    # keep the first row
    qwen_output = qwen_output.split('\n')[0]
    if qwen_output.startswith('Lucy：'):
        pass
    # add "Lucy：" at the start
    else:
        qwen_output = 'Lucy：' + qwen_output
    return qwen_output


# check qwen model output format
def check_qwen_output(qwen_output):
    if len(qwen_output.strip().split('\n')) != 1:
        return False
    if not qwen_output.startswith('Lucy'):
        return False
    return True


def chat(history, dialog):
    xiaoming_prompt = xiaoming_template
    for idx, d in enumerate(dialog.strip().replace('\n\n','\n').split('\n')):
        if idx % 2 == 0:
            xiaoming_prompt += '<|im_start|>user\n' + d.replace('小明：','') + '<|im_end|>\n'
        else:
            xiaoming_prompt += '<|im_start|>assistant\n' + d.replace('Lucy：','') + '<|im_end|>\n'
    xiaoming_input = xiaoming_tokenizer(xiaoming_prompt, return_tensors='pt').to(xiaoming_model.device)
    with torch.no_grad():
        xiaoming_output = xiaoming_model.generate(**xiaoming_input, do_sample=True,
                                                  max_new_tokens=100,
                                                  temperature=1.,
                                                  eos_token_id=xiaoming_tokenizer.eos_token_id,
                                                  pad_token_id=xiaoming_tokenizer.eos_token_id)
    xiaoming_output = '小明：' + xiaoming_tokenizer.decode(xiaoming_output[0]).split('<|im_start|>user')[-1].replace('<|im_end|>','').strip()
    dialog += '\n\n' + xiaoming_output
    qwen_prompt = qwen_template.replace('{history}', history).replace('{dialog}', dialog)
    qwen_input = qwen_tokenizer(qwen_prompt, return_tensors='pt').to(qwen_model.device)
    # 这里结果可能不对，多生成几次
    output_flag = False
    for i in range(5):
        with torch.no_grad():
            qwen_output = qwen_model.generate(**qwen_input, do_sample=True,
                                                      max_new_tokens=100,
                                                      temperature=1.,
                                                      eos_token_id=xiaoming_tokenizer.eos_token_id,
                                                      pad_token_id=xiaoming_tokenizer.eos_token_id)
            qwen_output = qwen_tokenizer.decode(qwen_output[0]).split('<|im_start|>assistant')[1].replace('<|im_end|>','').strip()
            output_flag = check_qwen_output(qwen_output)
            if output_flag:
                dialog += '\n' + qwen_output
                break
            else:
                print('格式有误：{}'.format(qwen_output))
    if not output_flag:
        qwen_output = format_qwen_output(qwen_output)
        dialog += '\n' + qwen_output
    return dialog


# load test data
episodes = []
with open(config['data_pathes']['test']) as f:
    for line in f:
        data = json.loads(line)
        output_data = {'callback_day':data['callback_day'],'dialog':[], 'topic':[], 'sub_topic':[]}
        flag = 1
        for k in data:
            # remove today's dialog
            if k != 'callback_day' and k != 'day{}'.format(len(data)-1):
                try:
                    topic, sub_topic = re.findall('主题：(.*)\n细化主题：(.*)\n\n', data[k])[0]
                except:
                    print(len(data), k, data[k])
                    flag = 0
                    break
                dialog = re.sub('主题.*\n细化主题.*\n\n', '', data[k])
                output_data['topic'].append(topic)
                output_data['sub_topic'].append(sub_topic)
                output_data['dialog'].append(dialog)
            elif k == 'day{}'.format(len(data)-1):
                start = '\n\n'.join(data[k].split('\n\n')[:2])
                gpt_ans = '\n\n'.join(data[k].split('\n\n')[2:])
                output_data['start'] = start
                output_data['gpt_ans'] = gpt_ans
        if flag:
            episodes.append(output_data)

print('total test data size:{}'.format(len(episodes)))

# search once
round = 10
for episode in tqdm(episodes):

    dialog = episode['start']
    sub_topic = episode['sub_topic']
    rank, score = embedding_retrieval(dialog, sub_topic, bge_model, bge_tokenizer)

    top_index = rank.index(min(rank))
    history = episode['dialog'][top_index]

    for r in range(round):
        dialog = chat(history, dialog)
    episode['search_once_ans'] = dialog

# search ecah round
for episode in tqdm(episodes):

    dialog = episode['start']
    sub_topic = episode['sub_topic']
    for r in range(round):
        rank, score = embedding_retrieval(dialog, sub_topic, bge_model, bge_tokenizer)
        top_index = rank.index(min(rank))
        print(rank)
        history = episode['dialog'][top_index]
        dialog = chat(history, dialog)
    episode['search_each_round_ans'] = dialog


with open('data/qwen_bge_result.json','w',encoding='utf-8') as f:
    f.write(json.dumps(episodes, ensure_ascii=False))