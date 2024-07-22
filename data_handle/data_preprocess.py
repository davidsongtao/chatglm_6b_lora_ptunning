import json
import traceback  # 返回的字符串包含有关异常的详细信息
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from functools import partial
from glm_config import *


def convert_samples(
        samples: dict,
        tokenizer,
        max_source_seq_len,
        max_target_seq_len
):
    """
    将样本数据转换为Prompt-Tunning模型接受的输入数据
    Args:
        samples(dict): 训练数据样本
        e.g. -> {
                    'text': [
                        '{"context"： “年基准利率...”, “target”: "2017年银行贷款基准..."}'
                    ]
                }
        max_source_seq_len(int): prompt最大长度
        max_target_seq_len(int): 答案最大长度

    Returns:
        dict(str: np.array) -> tokenized_output = {
                                            'input_ids': [[1325, 10...], [758, 2345, ...]...],
                                            'labels': [[8822, 10...], [125, 58...]...]
                                        }
    """
    # 定义输出格式
    tokenized_output = {
        'input_ids': [],
        'labels': []
    }

    # 设定最大句子长度
    max_seq_len = max_source_seq_len + max_target_seq_len

    #
    for sample in samples['text']:
        try:
            # 将json结构的数据样本text读取为字典格式
            sample = json.loads(sample)
            # 获取context和target
            context = sample['context']
            # print(f'context -> {context}')
            target = sample['target']
            # 对context进行编码
            prompt_ids = tokenizer.encode(text=context, add_special_tokens=False)
            print(f"prompt_ids -> {prompt_ids}")
            # 对target进行编码
            target_ids = tokenizer.encode(text=target, add_special_tokens=False)

            # 如果prompt的长度大于设定的最大长度，直接截断
            if len(prompt_ids) >= max_source_seq_len:
                prompt_ids = prompt_ids[:max_source_seq_len-1]
            # 如果target的长度大于设定的最大长度，直接截断
            if len(target_ids) >= max_target_seq_len:
                target_ids = target_ids[:max_target_seq_len-2]

            # 将prompt和target进行合并，作为模型的输入
            input_ids = tokenizer.build_inputs_with_special_tokens(prompt_ids, target_ids)
            print(len(prompt_ids))
            print(len(target_ids))
            print(len(input_ids))
            print(f"input_ids -> {input_ids}")



            break
        except:
            print(f'"{sample}" -> {traceback.format_exc()}')
            continue


if __name__ == '__main__':
    # 实例化配置文件
    param = ParametersConfig()
    # 加载训练数据
    train_dataset = load_dataset(path='text', data_files={'train': param.train_path})
    # print(train_dataset)
    # print('=' * 80)
    # print(train_dataset['train'])
    # print('=' * 80)
    # print(train_dataset['train']['text'])
    tokenizer = AutoTokenizer.from_pretrained(param.pretrained_model, trust_remote_code=True, revision='main')
    convert_samples(train_dataset['train'], tokenizer, max_source_seq_len=30, max_target_seq_len=20)
