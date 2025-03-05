#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file: ranker_utils.py
@time: 2024/5/21 10:07
@desc: 
"""
import torch


def get_embedding(model, tokenizer, sent):
    if isinstance(sent, str):
        sent = [sent]
    encoded_input = tokenizer(sent, padding=True, truncation=True, return_tensors='pt').to(model.device)
    with torch.no_grad():
        model_output = model(**encoded_input)
        # 看一下shape
        sentence_embedding = model_output[0][:,0]
    sentence_embedding = torch.nn.functional.normalize(sentence_embedding, p=2, dim=1)

    return sentence_embedding


def embedding_retrieval(query, answers, bge_model, bge_tokenizer):
    query_emb = get_embedding(bge_model, bge_tokenizer, query)
    answers_emb = get_embedding(bge_model, bge_tokenizer, answers)

    scores = query_emb @ answers_emb.T
    rank = torch.argsort(scores, descending=True).squeeze().tolist()
    return rank


def rank_model_retrieval(query, answers, rank_model, rank_tokenizer):
    sents = []
    # 这里的query 和 answers 格式都要是处理好的

    for answer in answers:
        sents.append(query + answer)

    scores = []
    stride = 3
    for i in range(0, len(sents), stride):
        inputs = rank_tokenizer(sents[i:i + stride], padding='longest', return_tensors='pt').to(
            rank_model.pretrained_model.device)

        with torch.no_grad():
            _, _, values = rank_model(**inputs, output_hidden_states=True, return_dict=True)

        for i in range(values.shape[0]):
            # use the score on the last token except pad token for inference
            chosen_length = (inputs['input_ids'][i] != rank_tokenizer.pad_token_id).nonzero()[-1] + 1
            scores.append(values[i, chosen_length - 1].item())

    rank = torch.argsort(torch.tensor(scores), descending=True).squeeze().tolist()
    return rank

