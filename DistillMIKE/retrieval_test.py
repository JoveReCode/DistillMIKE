import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import GPTJForCausalLM, GPT2Tokenizer
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from transformers import set_seed
# from transformers import GPT2Tokenizer, OPTForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

import json
import argparse
import random
import pickle
import os

def parse_args():
    parser = argparse.ArgumentParser(description="In Context Learning for pretrained GPTs")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args
dataset_name = './multi_counterfact_ori.json'

# test_num = 20000
# idx = 0
# icl_examples = []
# count=0
# with open('corpus_idx_retrieval2.txt', 'r') as fIn:
#     lines = fIn.readlines()
#     lines = [line[:-1] for line in lines]
#     corpus_idx = [[int(idx) for idx in line.split()] for line in lines]
#
# acc=1
# for i,idx in enumerate(corpus_idx):
#     if i % 2 == 0:
#
#         if idx[0] == (20000+int(i/2)):
#             count += 1
#     else:
#
#         if idx[0] == int(20000+i-acc):
#             count += 1
#         acc+=1
# print(count)

# with open(dataset_name, 'r') as f:
#     lines = json.load(f)
# for i, line in enumerate(lines):
#
#     relation = line['requested_rewrite']['relation_id']
#     prompt = line['requested_rewrite']['prompt']
#     subject = line['requested_rewrite']['subject']
#     prompt_calibrate = prompt.format('SUBJECT')
#     prompt = prompt.format(subject)
#     PROMPTS = [prompt, prompt_calibrate]
#
#     target_true = line['requested_rewrite']['target_true']['str']
#     target_new = line['requested_rewrite']['target_new']['str']
#
#     PPLs = []
#     targets = [target_new, target_true]
#     if i == 3435:
#         print(line)

score_list=[]
with open('retrieval_score_n.txt', 'r') as fIn:
    lines = fIn.readlines()
    for line in lines:
        # print(line)
        score_list.append(float(line))

    score_list.sort()
    print(score_list[-10000:])

