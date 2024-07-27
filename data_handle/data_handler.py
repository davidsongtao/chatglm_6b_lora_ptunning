"""
Description: 该脚本用于将样本数据转换成模型接受的输入格式。
    
-*- Encoding: UTF-8 -*-
@File     ：data_handler.py
@Author   ：King Songtao
@Time     ：2024/7/27 下午3:10
@Contact  ：king.songtao@gmail.com
"""
import json

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from glm_config import *
from datasets import load_dataset


def convert_samples(samples: dict, tokenizer, max_context_len, max_target_len):
    """
    Args:
        samples: 原始数据样本，格式为字典
            e.g. -> samples = {
                            "text": [
                                {
                                    'context': 'Instruction: 你现在是一个...',
                                    'target': '```json\n...'
                                },
                                {
                                    'context': 'Instruction: 你现在是一个...',
                                    'target': '```json\n...'
                                },
                                ...
                            ]
                        }
        tokenizer: 分词器
        max_context_len: 提示词部分最大长度
        max_target_len: 真实标签部分最大长度

    Returns:
        tokenized_input = {
            'input_ids': [[1231,2345,234,...]...],
            'labels': [[1231,2345,234,...]...]
        }
    """

    tokenized_input = {
        'input_ids': [],
        'labels': []
    }

    max_sq_len = max_context_len + max_target_len

    for sample in samples['text']:
        sample = json.loads(sample)

        context = sample['context']
        target = sample['target']

        context = tokenizer.encode(context, add_special_tokens=False)
        target = tokenizer.encode(target, add_special_tokens=False)

        if len(context) >= max_context_len:
            context = context[:max_context_len - 1]
        if len(target) >= max_target_len:
            target = target[:max_target_len - 2]

        input_ids = tokenizer.build_inputs_with_special_tokens(context, target)
        sop_index = input_ids.index(tokenizer.bos_token_id)
        labels = [-100] * sop_index + input_ids[sop_index:]

        padding_len = max_sq_len - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_len
        labels += [-100] * padding_len

        tokenized_input['input_ids'].append(input_ids)
        tokenized_input['labels'].append(labels)

    for k, v in tokenized_input.items():
        tokenized_input[k] = np.array(v)

    return tokenized_input


def get_max_length(tokenizer, dataset_file: str):
    """
    测试数据集最大的输入/输出tokens是多少

    Args:
        dataset_file(str): _description_
    """
    source_seq_len_list = []
    target_seq_len_list = []

    with open(dataset_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            line = json.loads(line)

            source_len = len(tokenizer.encode(line['context']))
            source_seq_len_list.append(source_len)

            target_len = len(tokenizer.encode(line['target']))
            target_seq_len_list.append(target_len)

        print(dataset_file)
        print("=" * 80)
        print(f'[Source Sequence] Max: {max(source_seq_len_list)}')
        print(f'[Source Sequence] Avg: {int(sum(source_seq_len_list) / len(source_seq_len_list))}')
        print(f'[Source Sequence] Mid: {sorted(source_seq_len_list)[int(len(source_seq_len_list) / 2)]}')
        print("=" * 80)
        print(f'[Target Sequence] Max: {max(target_seq_len_list)}')
        print(f'[Target Sequence] Avg: {int(sum(target_seq_len_list) / len(target_seq_len_list))}')
        print(f'[Target Sequence] Mid: {sorted(target_seq_len_list)[int(len(target_seq_len_list) / 2)]}')


if __name__ == '__main__':
    param = ParametersConfig()
    train_dataset = load_dataset(path='text', data_files={'train': param.train_path, 'dev': param.eval_path})
    train_dataset = train_dataset['train']
    tokenizer = AutoTokenizer.from_pretrained(param.pretrained_model, trust_remote_code=True, revision='main')
    train_dataset = convert_samples(train_dataset, tokenizer, max_context_len=300, max_target_len=250)
    print(train_dataset)
