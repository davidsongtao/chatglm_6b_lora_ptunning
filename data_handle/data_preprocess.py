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
            # print(f"prompt_ids -> {prompt_ids}")
            # 对target进行编码
            target_ids = tokenizer.encode(text=target, add_special_tokens=False)

            # 如果prompt的长度大于设定的最大长度，直接截断(-1是为了存放130001)
            if len(prompt_ids) >= max_source_seq_len:
                prompt_ids = prompt_ids[:max_source_seq_len - 1]
            # 如果target的长度大于设定的最大长度，直接截断（-2是为了存放130004和130005）
            if len(target_ids) >= max_target_seq_len:
                target_ids = target_ids[:max_target_seq_len - 2]

            # 将prompt和target进行合并，作为模型的输入
            input_ids = tokenizer.build_inputs_with_special_tokens(prompt_ids, target_ids)
            # print(len(prompt_ids))
            # print(len(target_ids))
            # print(len(input_ids))
            # print(f"input_ids -> {input_ids}")

            # 从input_ids中取出labels
            # 1. 计算context文本的长度
            context_len = input_ids.index(tokenizer.bos_token_id)
            # 2. 计算掩码的位置
            mask_position = context_len - 1
            # 将context部分填充为-100，target部分保留，拼装成labels
            labels = [-100] * context_len + input_ids[mask_position + 1:]
            # print(labels)
            # print(len(labels))

            # 确保input_ids不超过句子最大长度， 如果句子太短，要进行填充(pad_len->需要补齐多少位)
            pad_len = max_seq_len - len(input_ids)
            # print(pad_len)
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len

            tokenized_output['input_ids'].append(input_ids)
            tokenized_output['labels'].append(labels)


        except:
            print(f'"{sample}" -> {traceback.format_exc()}')
            continue

    # 遍历tokenized_output, 将值改变为np array形式（map这些数值的时候必须是数组）
    for k, v in tokenized_output.items():
        tokenized_output[k] = np.array(v)

    return tokenized_output


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
    tokenized_output = convert_samples(train_dataset['train'], tokenizer, max_source_seq_len=200, max_target_seq_len=100)

