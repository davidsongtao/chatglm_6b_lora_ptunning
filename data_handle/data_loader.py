"""
Description: 该脚本用于通过处理过的dataset创建dataloader,为模型输入做准备
    
-*- Encoding: UTF-8 -*-
@File     ：data_loader.py
@Author   ：King Songtao
@Time     ：2024/7/27 下午5:46
@Contact  ：king.songtao@gmail.com
"""
from torch.utils.data import DataLoader
from glm_config import *
from transformers import AutoTokenizer, default_data_collator
from datasets import load_dataset
from functools import partial
from data_handle.data_handler import *


def get_dataloader(train_path, eval_path):

    _tokenizer = AutoTokenizer.from_pretrained(param.pretrained_model, trust_remote_code=True, revision='main')
    _dataset = load_dataset(path='text', data_files={'train': train_path, 'eval': eval_path})

    new_func = partial(
        convert_samples,
        tokenizer=_tokenizer,
        max_context_len=param.max_source_sq_len,
        max_target_len=param.max_target_sq_len
    )

    _dataset = _dataset.map(new_func, batched=True)

    _train_dataset = _dataset['train']
    _eval_dataset = _dataset['eval']

    _train_dataloader = DataLoader(
        _train_dataset,
        batch_size=param.batch_size,
        shuffle=True,
        collate_fn=default_data_collator
    )

    _eval_dataloader = DataLoader(
        _eval_dataset,
        batch_size=param.batch_size,
        shuffle=True,
        collate_fn=default_data_collator
    )

    return _train_dataloader, _eval_dataloader


if __name__ == '__main__':
    param = ParametersConfig()
    train_dataloader, eval_dataloader = get_dataloader(param.train_path, param.eval_path)
    for i, v in enumerate(train_dataloader):
        print(i)
        print(v['labels'])
        break
