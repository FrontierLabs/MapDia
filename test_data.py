#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file: utils.py
@time: 2024/5/21 9:39
@desc: 
"""

import re
import json

from utils import load_config

def load_episodes(config):
    with open(config['data_pathes']['episodes']) as f:
        episodes = json.load(f)
    return episodes


def preprocess_dialog(config):
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

    with open(config['data_pathes']['episodes'], "w") as f:
        json.dump(episodes, f)


if __name__ == '__main__':
    config = load_config()
    preprocess_dialog(config)
