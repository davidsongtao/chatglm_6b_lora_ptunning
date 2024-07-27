"""
Description: 该脚本主要实现模型训练。将准备好的数据集送入模型进行训练。
    
-*- Encoding: UTF-8 -*-
@File     ：train.py
@Author   ：King Songtao
@Time     ：2024/7/27 下午8:05
@Contact  ：king.songtao@gmail.com
"""
from transformers import AutoTokenizer, AutoModel
from glm_config import *


def model2train():
    _param = ParametersConfig()
    tokenizer = AutoTokenizer.from_pretrained(_param.pretrained_model, trust_remote_code=True, revision='main')
    print("ok")


if __name__ == '__main__':
    model2train()
