#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file: qwen_rank.py
@time: 2024/5/21 9:26
@desc: 
"""

import json

import torch
from tqdm import tqdm

from ranker_utils import rank_model_retrieval
from utils import load_rank_model, load_qwen, load_config
from test_data import load_episodes


config = load_config()

def _add_or_replace_eos_token(tokenizer, eos_token) -> None:
    is_added = tokenizer.eos_token_id is None
    is_out_of_vocabulary = eos_token not in tokenizer.get_vocab()
    tokenizer.add_special_tokens({"eos_token": eos_token})

    if is_added:
        print("Add eos token: {}".format(tokenizer.eos_token))
    else:
        print("Replace eos token: {}".format(tokenizer.eos_token))

    if is_out_of_vocabulary:
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

qwen_device = torch.device('cuda:1')
qwen_path = config['model_pathes']['qwen']
qwen_model, qwen_tokenizer = load_qwen(qwen_path, qwen_device)

_add_or_replace_eos_token(qwen_tokenizer, '<|im_end|>')

with open('data/episodes.json') as f:
    episodes = json.load(f)

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

template = """<|im_start|>system
你是一个对话标题打分助手，针对给定的对话与标题，依据其相关程度进行打分。<|im_end|>
<|im_start|>user
{dialog}<|im_start|>assistant
{sub_topic}<|im_end|>"""


# 没有tuned的qwen传入的结果可能不
def format_qwen_output(qwen_output):
    # 这里切开只要第一行，如果不是Lucy：开头，则补上
    qwen_output = qwen_output.split('\n')[0]
    if qwen_output.startswith('Lucy：'):
        pass
    else:
        qwen_output = 'Lucy：' + qwen_output
    return qwen_output


def check_qwen_output(qwen_output):
    if len(qwen_output.strip().split('\n')) != 1:
        return False
    if not qwen_output.startswith('Lucy'):
        return False
    return True


def chat(history, dialog, greedy=False):
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
    qwen_prompt = qwen_template.replace('{history}', history).replace('{dialog}', dialog)
    qwen_input = qwen_tokenizer(qwen_prompt, return_tensors='pt').to(qwen_model.device)
    # 这里结果可能不对，多生成几次
    output_flag = False
    for i in range(5):
        with torch.no_grad():
            qwen_output = qwen_model.generate(**qwen_input, do_sample=True,
                                              max_new_tokens=100,
                                              temperature=1.,
                                              eos_token_id=qwen_tokenizer.eos_token_id,
                                              pad_token_id=qwen_tokenizer.eos_token_id)
            decoded_output = qwen_tokenizer.decode(qwen_output[0])
            qwen_output = decoded_output.split('<|im_start|>assistant')[1].replace('<|im_end|>', '').strip()
            output_flag = check_qwen_output(qwen_output)
            if output_flag:
                dialog += '\n' + qwen_output
                break
            else:
                #                 print('格式有误：{}'.format(qwen_output))
                continue
    if not output_flag:
        qwen_output = format_qwen_output(qwen_output)
        dialog += '\n' + qwen_output
    return dialog


def format_history(history, topic, sub_topic):
    history = '主题：{}\n细化主题：{}\n\n'.format(topic, sub_topic) + history
    return history


episodes = load_episodes(config)

## 检索一次
round = 10
for episode in tqdm(episodes):

    dialog = episode['start']
    topic = episode['topic']
    sub_topic = episode['sub_topic']
    rank = rank_model_retrieval(dialog, sub_topic, rank_model, rank_tokenizer)

    top_index = rank[0]
    history = episode['dialog'][top_index]
    history = format_history(history, topic[top_index], sub_topic[top_index])
    episode['retrieve_once'] = [top_index]
    for r in range(round):
        if r == 0:
            dialog = chat(history, dialog, greedy=True)
        else:
            dialog = chat(history, dialog)
    episode['search_once_ans'] = dialog
    torch.cuda.empty_cache()

## 检索每轮
round = 10
for episode in tqdm(episodes):

    dialog = episode['start']
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
            dialog = chat(history, dialog, greedy=True)
        else:
            dialog = chat(history, dialog)
    episode['search_each_round_ans'] = dialog
    torch.cuda.empty_cache()

with open('data/qwen_rank_result.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(episodes, ensure_ascii=False))
