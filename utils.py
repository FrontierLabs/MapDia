#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file: utils.py
@time: 2024/5/21 9:39
@desc: 
"""
import torch
import yaml

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead


CONFIG_PATH = "config.yaml"


def load_bge(bge_path, bge_device):
    # bert model不支持device_map='auto'
    bge = AutoModel.from_pretrained(bge_path)
    bge_tokenizer = AutoTokenizer.from_pretrained(bge_path)
    bge.to(bge_device)
    bge.eval()
    return bge, bge_tokenizer


def load_qwen(qwen_path, device=None):
    if device:
        qwen = AutoModelForCausalLM.from_pretrained(qwen_path, torch_dtype=torch.float16, device_map=device)
    else:
        qwen = AutoModelForCausalLM.from_pretrained(qwen_path, torch_dtype=torch.float16, device_map='auto')
    qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_path, use_fast=False)
    qwen.eval()
    return qwen, qwen_tokenizer


def load_rank_model(rank_model_path, vhead_file, rank_device):
    rank_model = AutoModelForCausalLMWithValueHead.from_pretrained(rank_model_path)
    rank_tokenizer = AutoTokenizer.from_pretrained(rank_model_path, use_fast=False)

    # 这里要把value head权重加载进来
    from safetensors import safe_open
    with safe_open(vhead_file, framework="pt", device="cpu") as f:
        vhead_param = {key: f.get_tensor(key) for key in f.keys()}

    rank_model.load_state_dict(vhead_param, strict=False)
    rank_model = rank_model.half()
    rank_model.eval()
    rank_model.to(rank_device)

    return rank_model, rank_tokenizer


def load_config(path = CONFIG_PATH):
    """
    Loads and parses a YAML configuration file.
    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg
