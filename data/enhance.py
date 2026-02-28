# -*- coding: utf-8 -*-
# Time: 2026-2-27 ~ now
# Create by: 程惠泽
# Email: hzcheng@chd.edu.cn
# Created by: Visual Studio Code 1.104.0
# 使用三种方式对 QA 进行数据增强： 1. BERT 随机插入（100%）  2.同义词替换（100%）  3. BERT 上下文替换（20%）

# 输入 gt.json
# 输出 gt_enhance.json

# 导包
import torch
torch.__version__ = "2.6.0" 
import os
import json
os.environ["HF_SKIP_TORCH_LOAD_SAFETY_CHECK"] = "True"
import transformers.utils.import_utils as import_utils
import transformers.modeling_utils as modeling_utils
def dummy_check(*args, **kwargs):
    return True
import_utils.check_torch_load_is_safe = dummy_check
modeling_utils.check_torch_load_is_safe = dummy_check
from transformers import BertTokenizer, PreTrainedTokenizerBase
if not hasattr(BertTokenizer, '_convert_token_to_id'):
    BertTokenizer._convert_token_to_id = BertTokenizer.convert_tokens_to_ids
if not hasattr(PreTrainedTokenizerBase, '_convert_token_to_id'):
    PreTrainedTokenizerBase._convert_token_to_id = PreTrainedTokenizerBase.convert_tokens_to_ids
import nlpaug.augmenter.word as naw
import random
from tqdm import tqdm
import copy


if __name__ == '__main__':
    # 1. 获取原数据集
    with open('data/gt.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 2. 初始化数据增强对象
    aug_ins = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert", device='cpu')
    aug_syn = naw.SynonymAug(aug_src='wordnet')
    aug_bert = naw.ContextualWordEmbsAug(model_path='Model/bert-base-uncased', action="substitute", device='cpu')

    # 3. 对每一个样例进行文本数据增强
    ans = []
    for item in tqdm(data, colour='green'):
        ans.append(copy.deepcopy(item))  # 在原始数据集的基础上进行增加
        text = item["conversations"][0]["value"].split("Options")[0][16:]  # 待增强的文本

        # BERT 随机插入（100%）
        augmented_text = aug_ins.augment(text)[0]
        item["conversations"][0]["value"] = '<image>Question: ' + augmented_text + 'Options' + item["conversations"][0]["value"].split("Options")[1]
        ans.append(copy.deepcopy(item))

        # 同义词替换（100%）
        augmented_text = aug_syn.augment(text)[0]
        item["conversations"][0]["value"] = '<image>Question: ' + augmented_text + 'Options' + item["conversations"][0]["value"].split("Options")[1]
        ans.append(copy.deepcopy(item))

        # BERT 上下文替换（20%）
        if random.random() < 0.2:
            augmented_text = aug_bert.augment(text)[0]
            item["conversations"][0]["value"] = '<image>Question: ' + augmented_text + 'Options' + item["conversations"][0]["value"].split("Options")[1]
            ans.append(copy.deepcopy(item))

    print(len(ans))
    with open('data/gt_enhance.json', 'w', encoding='utf-8') as f:
        json.dump(ans, f , ensure_ascii=False, indent=4)
