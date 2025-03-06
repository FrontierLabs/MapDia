#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file: all_in_one_gpt4.py
@time: 2024/5/21 9:29
@desc: 
"""
import json
import re

import torch
from gpt_with_temperature import get_gpt_response
from tqdm import tqdm

from utils import load_qwen, load_config
from test_data import load_episodes


config = load_config()

xiaoming_path = config['model_pathes']['user_model']
device = torch.device('cuda:0')
xiaoming_model, xiaoming_tokenizer = load_qwen(xiaoming_path)


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


_add_or_replace_eos_token(xiaoming_tokenizer, '<|im_end|>')

xiaoming_template = """<|im_start|>system
你是小明，正在与对话机器人Lucy进行聊天。<|im_end|>
"""

gpt_template_history = """你是机器人Lucy，正在与小明进行聊天，你的任务是依据给定的对话历史与当前对话，对小明进行回复。
对话历史：
###
"""

gpt_template_dialog = """当前对话：
{}
结合对话历史与当前对话，给出Lucy的回复。回复分为两部分：
1.Thoughts：判断当前对话与哪一天历史对话可能有潜在联系，最后判断是否可以将对话主题转移到历史对话主题上，如果话题联系度不高，则不能将话题转移到历史对话上。输出yes或者no。
2.Lucy回复：这一部分输出Lucy正常的回复。

一下是一个回复示例，按照这个格式回复：
Thoughts：当前对话提及了跑步，可能和历史对话小明参加马拉松比赛有关，可以将对话转移到历史对话中。yes
Lucy：小明，说起跑步，上次你参加马拉松比赛怎么样呀？

给出Lucy的回复"""


def check_gpt_output(callback_output):
    try:
        thoughts, dialog = callback_output.split('\n')
    except:
        return False
    if not thoughts.startswith('Thoughts'):
        return False
    if not dialog.startswith('Lucy'):
        return False
    return True


def chat(history_prompt, dialog, dialog_with_thoughts, greedy=False):
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
    # 这里结果可能不对，多生成几次
    output_flag = False
    prompt = history_prompt + gpt_template_dialog.format(dialog)
    for i in range(3):
        gpt_output = get_gpt_response(prompt, "118", temperature=1.0)
        if check_gpt_output(gpt_output):
            break
    dialog_with_thoughts += '\n' + gpt_output
    # remove thoughts
    try:
        dialog += '\n' + gpt_output.split('\n')[1]
    except:
        dialog += '\n' + gpt_output
    return dialog, dialog_with_thoughts


def format_history(history, topic, sub_topic):
    history = '主题：{}\n细化主题：{}\n\n'.format(topic, sub_topic) + history
    return history


episodes = load_episodes(config)

## 检索一次
torch.cuda.empty_cache()
round = 10
select_days = 6
for i, episode in tqdm(enumerate(episodes)):
    if not episode['gpt_dialog_with_thoughts']:
        print(i)
        round = 7
    else:
        continue
    dialog = dialog_with_thoughts = episode['start']
    topic = episode['topic']
    sub_topic = episode['sub_topic']
    history_prompt = gpt_template_history
    try:
        # 先拿出callback day的数据：
        if len(episode['dialog']) >= select_days:
            callback_day = episode['callback_day']
            callback_idx = int(re.findall('day(.*)', callback_day)[0]) - 1
            history = episode['dialog'][callback_idx]
            history = format_history(history, topic[callback_idx], sub_topic[callback_idx])
            history_prompt += 'day{}:\n'.format(idx + 1) + history + '\n###\n'
        else:
            callback_day = None
        for idx in range(min(len(episode['dialog']), 7)):
            if idx - 1 == callback_day:
                continue
            history = episode['dialog'][idx]
            history = format_history(history, topic[idx], sub_topic[idx])
            history_prompt += 'day{}:\n'.format(idx + 1) + history + '\n###\n'
        for r in range(round):
            if r == 0:
                dialog, dialog_with_thoughts = chat(history_prompt, dialog, dialog_with_thoughts, greedy=True)
            else:
                dialog, dialog_with_thoughts = chat(history_prompt, dialog, dialog_with_thoughts)
    except:
        print(i)
        dialog_with_thoughts = ''
    episode['gpt_dialog_with_thoughts'] = dialog_with_thoughts
    torch.cuda.empty_cache()

with open('data/gpt_result.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(episodes, ensure_ascii=False))
