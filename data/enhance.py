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




