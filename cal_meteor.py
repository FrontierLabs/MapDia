#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file: cal_meteor.py
@time: 2024/6/5 17:05
@desc: 
"""

from nltk.translate.meteor_score import meteor_score
import nltk
import json


nltk.download('wordnet')

with open('tuned_ans.json', encoding='utf-8') as f:
    model_ans = json.load(f)

with open('not_tuned_ans.json', encoding='utf-8') as f:
    not_tuned_ans = json.load(f)

with open('summary_ans.json', encoding='utf-8') as f:
    summary_ans = json.load(f)

all_score = []
for ans, truth in zip(model_ans, summary_ans):
    sub_topic, ground_truth = ' '.join(list(ans[1])), ' '.join(list(truth[1]))
    score = meteor_score([ground_truth], sub_topic)
    all_score.append(score)

print(sum(all_score)/len(all_score))


all_score = []
for ans, truth in zip(not_tuned_ans, summary_ans):
    sub_topic, ground_truth = ' '.join(list(ans)), ' '.join(list(truth[1]))
    score = meteor_score([ground_truth], sub_topic)
    all_score.append(score)

print(sum(all_score)/len(all_score))