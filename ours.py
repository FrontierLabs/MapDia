#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file: ours.py
@time: 2024/5/21 9:28
@desc: 
"""
import json

import torch
from tqdm import tqdm

from ranker_utils import rank_model_retrieval
from utils import load_rank_model, load_qwen, load_config

config = load_config


def _add_or_replace_eos_token(tokenizer: "PreTrainedTokenizer", eos_token: str) -> None:
    is_added = tokenizer.eos_token_id is None
    is_oov = eos_token not in tokenizer.get_vocab()
    tokenizer.add_special_tokens({"eos_token": eos_token})

    if is_added:
        print("Add eos token: {}".format(tokenizer.eos_token))
    else:
        print("Replace eos token: {}".format(tokenizer.eos_token))

    if is_oov:
        print("New tokens have been added, make sure `resize_vocab` is True.")


rank_device = torch.device('cuda:3')
rank_path = config['model_pathes']['ranker']['model']
vhead_file = config['model_pathes']['ranker']['vhead']

rank_model, rank_tokenizer = load_rank_model(rank_path, vhead_file, rank_device)

xiaoming_device = torch.device('cuda:2')

xiaoming_path = config['model_pathes']['user_model']
xiaoming_model, xiaoming_tokenizer = load_qwen(xiaoming_path, xiaoming_device)

# xiaoming_tokenizer没有设置<|im_end|>为eos_token
_add_or_replace_eos_token(xiaoming_tokenizer, '<|im_end|>')

callback_device = torch.device('cuda:1')
callback_path = config['model_pathes']['callback']
callback_model, callback_tokenizer = load_qwen(callback_path, callback_device)

with open('data/episodes.json') as f:
    episodes = json.load(f)

xiaoming_template = """<|im_start|>system
你是小明，正在与对话机器人Lucy进行聊天。<|im_end|>
"""

# callback的history里面是有topic和sub_topic的，要注意还原回去
callback_template = """<|im_start|>system
你是Lucy，与小明进行聊天，依据历史对话和当前对话，进行回复。聊天的过程中，需要结合给定历史对话，考虑是否可以将当前对话主题转移到历史对话主题，时机的判断以Thoughts形式展示。<|im_end|>
<|im_start|>user
对话历史如下：
###
{history}
###
当前对话如下：
###
{dialog}
###
<|im_end|>
<|im_start|>assistant"""

template = """<|im_start|>system
你是一个对话标题打分助手，针对给定的对话与标题，依据其相关程度进行打分。<|im_end|>
<|im_start|>user
{dialog}<|im_start|>assistant
{sub_topic}<|im_end|>"""


def check_callback_output(callback_output):
    try:
        thoughts, dialog = callback_output.split('\n')
    except:
        return False
    if not thoughts.startswith('Thoughts'):
        return False
    if not dialog.startswith('Lucy'):
        return False
    return True


def chat(history, dialog, dialog_with_thoughts, greedy=False):
    xiaoming_prompt = xiaoming_template
    for idx, d in enumerate(dialog.strip().replace('\n\n', '\n').split('\n')):
        if idx % 2 == 0:
            xiaoming_prompt += '<|im_start|>user\n' + d.replace('小明：', '') + '<|im_end|>\n'
        else:
            xiaoming_prompt += '<|im_start|>assistant\n' + d.replace('Lucy：', '') + '<|im_end|>\n'
    xiaoming_input = xiaoming_tokenizer(xiaoming_prompt, return_tensors='pt').to(xiaoming_model.device)
    with torch.no_grad():
        if greedy:
            xiaoming_output = xiaoming_model.generate(**xiaoming_input, do_sample=False,
                                                      max_new_tokens=100,
                                                      eos_token_id=xiaoming_tokenizer.eos_token_id,
                                                      pad_token_id=xiaoming_tokenizer.eos_token_id)
        else:
            xiaoming_output = xiaoming_model.generate(**xiaoming_input, do_sample=True,
                                                      max_new_tokens=100,
                                                      temperature=1.,
                                                      eos_token_id=xiaoming_tokenizer.eos_token_id,
                                                      pad_token_id=xiaoming_tokenizer.eos_token_id)
    xiaoming_output = '小明：' + xiaoming_tokenizer.decode(xiaoming_output[0]).split('<|im_start|>user')[-1].replace(
        '<|im_end|>', '').strip()
    dialog += '\n\n' + xiaoming_output
    dialog_with_thoughts += '\n\n' + xiaoming_output

    callback_prompt = callback_template.replace('{history}', history).replace('{dialog}', dialog)
    callback_input = callback_tokenizer(callback_prompt, return_tensors='pt').to(callback_model.device)

    # 如果失败，最多生成三次
    for i in range(3):
        with torch.no_grad():
            callback_output = callback_model.generate(**callback_input, do_sample=True,
                                                      max_new_tokens=250,
                                                      temperature=1.,
                                                      eos_token_id=xiaoming_tokenizer.eos_token_id,
                                                      pad_token_id=xiaoming_tokenizer.eos_token_id)
            callback_output = callback_tokenizer.decode(callback_output[0]).split('<|im_start|>assistant')[1].replace(
                '<|im_end|>', '').strip()
        if check_callback_output(callback_output):
            break
        else:
            print(callback_output)

    dialog_with_thoughts += '\n' + callback_output
    # remove thoughts
    try:
        dialog += '\n' + callback_output.split('\n')[1]
    except:
        dialog += '\n' + callback_output
    return dialog, dialog_with_thoughts


def format_history(history, topic, sub_topic):
    history = '主题：{}\n细化主题：{}\n\n'.format(topic, sub_topic) + history
    return history


## 检索一次
round = 10
for episode in tqdm(episodes):

    dialog = dialog_with_thoughts = episode['start']
    topic = episode['topic']
    sub_topic = episode['sub_topic']
    rank = rank_model_retrieval(dialog, sub_topic, rank_model, rank_tokenizer)

    top_index = rank[0]
    history = episode['dialog'][top_index]
    history = format_history(history, topic[top_index], sub_topic[top_index])
    episode['retrieve_once'] = [top_index]
    for r in range(round):
        if r == 0:
            dialog, dialog_with_thoughts = chat(history, dialog, dialog_with_thoughts, greedy=True)
        else:
            dialog, dialog_with_thoughts = chat(history, dialog, dialog_with_thoughts)
    episode['search_once_dialog_with_thoughts'] = dialog_with_thoughts
    torch.cuda.empty_cache()

## 检索每轮
round = 10
for episode in tqdm(episodes):

    dialog = dialog_with_thoughts = episode['start']
    topic = episode['topic']
    sub_topic = episode['sub_topic']

    episode['retrieve_each_round'] = []
    for r in range(round):
        rank = rank_model_retrieval(dialog, sub_topic, rank_model, rank_tokenizer)
        top_index = rank[0]
        episode['retrieve_each_round'].append(top_index)
        history = episode['dialog'][top_index]
        history = format_history(history, topic[top_index], sub_topic[top_index])
        if r == 0:
            dialog, dialog_with_thoughts = chat(history, dialog, dialog_with_thoughts, greedy=True)
        else:
            dialog, dialog_with_thoughts = chat(history, dialog, dialog_with_thoughts)
    episode['search_each_round_dialog_with_thoughts'] = dialog_with_thoughts
    torch.cuda.empty_cache()

with open('data/ours_result.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(episodes, ensure_ascii=False))
