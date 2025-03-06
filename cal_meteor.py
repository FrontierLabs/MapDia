#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Calculate METEOR scores between model generated texts and reference answers
Using NLTK's meteor_score to compute text similarity
"""

from nltk.translate.meteor_score import meteor_score
import nltk
import json

# Download WordNet dictionary required for METEOR scoring
nltk.download('wordnet')

# Load fine-tuned model outputs
with open('tuned_ans.json', encoding='utf-8') as f:
    model_ans = json.load(f)

# Load outputs from model without fine-tuning
with open('not_tuned_ans.json', encoding='utf-8') as f:
    not_tuned_ans = json.load(f)

# Load reference answers
with open('summary_ans.json', encoding='utf-8') as f:
    summary_ans = json.load(f)

# Calculate METEOR scores for fine-tuned model
all_score = []
for ans, truth in zip(model_ans, summary_ans):
    # Convert texts to space-separated string format
    sub_topic, ground_truth = ' '.join(list(ans[1])), ' '.join(list(truth[1]))
    score = meteor_score([ground_truth], sub_topic)
    all_score.append(score)

# Print average METEOR score for fine-tuned model
print(sum(all_score)/len(all_score))

# Calculate METEOR scores for model without fine-tuning
all_score = []
for ans, truth in zip(not_tuned_ans, summary_ans):
    sub_topic, ground_truth = ' '.join(list(ans)), ' '.join(list(truth[1]))
    score = meteor_score([ground_truth], sub_topic)
    all_score.append(score)

# Print average METEOR score for model without fine-tuning
print(sum(all_score)/len(all_score))
