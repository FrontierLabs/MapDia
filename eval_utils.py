#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file: eval_utils.py
@time: 2024/5/24 10:16
@desc: 
"""
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
from rouge import Rouge
import jieba


def load_qwen(qwen_path):
    qwen = AutoModelForCausalLM.from_pretrained(qwen_path, torch_dtype=torch.float16, device_map='auto')
    qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_path, use_fast=False)

    qwen.eval()
    return  qwen, qwen_tokenizer


def get_rouge_score(model_ans, summary_ans):
    rouger = Rouge()
    all_score = []
    for ans, truth in zip(model_ans, summary_ans):
        sub_topic, ground_truth = ' '.join(jieba.lcut(ans)), ' '.join(jieba.lcut(truth))

        score = rouger.get_scores(sub_topic, ground_truth)
        all_score.append(score)