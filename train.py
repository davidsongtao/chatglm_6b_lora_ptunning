"""
Description: 该脚本主要实现模型LoRA训练。将准备好的数据集送入模型进行训练。
    
-*- Encoding: UTF-8 -*-
@File     ：train.py
@Author   ：King Songtao
@Time     ：2024/7/27 下午8:05
@Contact  ：king.songtao@gmail.com
"""
import peft
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_scheduler
from glm_config import *
from torch.cuda.amp import autocast
from utils.common_utils import *
from data_handle.data_loader import *


def model2train():

    # 实例化需要用到的东西
    _param = ParametersConfig()
    _tokenizer = AutoTokenizer.from_pretrained(_param.pretrained_model, trust_remote_code=True, revision='main')
    _config = AutoConfig.from_pretrained(_param.pretrained_model, trust_remote_code=True, revision='main')
    _model = AutoModel.from_pretrained(_param.pretrained_model, config=_config, trust_remote_code=True, revision='main').half().cuda()
    _model.to(_param.device)

    # 使用梯度检查点优化，用于在反向传播中降低内存使用。只保存部分激活值，未保存的反向传播时重新计算。
    _model.gradient_checkpointing_enable()
    _model.enable_input_require_grads()

    # 缓存优化，设置不进行缓存
    _model.config.use_cache = False

    peft_config = peft.LoraConfig(
        peft_type=peft.TaskType.CAUSAL_LM,
        inference_mode=False,
        r=_param.lora_rank,
        lora_alpha=32,
        lora_dropout=0.1
    )

    # 通过ChatGLM模型和LoRA配置文件构建合并了LoRA的新模型
    _model = peft.get_peft_model(_model, peft_config)

    # 权重衰减，防止过拟合
    _no_decay = ['bias', 'LayerNorm.weight']
    _optimizer_grouped_parameters = [
        {
            "params": [p for n, p in _model.named_parameters() if not any(nd in n for nd in _no_decay)],
            "weight_decay": _param.weight_decay
        },
        {
            "params": [p for n, p in _model.named_parameters() if any(nd in n for nd in _no_decay)],
            "weight_decay": 0.0
        }
    ]

    # 构建优化器
    _optimizer = torch.optim.AdamW(_optimizer_grouped_parameters, lr=_param.learning_rate)
    _model.to(_param.device)

    # 准备数据集
    _train_dataloader, _eval_dataloader = get_dataloader(_param.train_path, _param.eval_path)

    # 学习率预热
    _max_steps_per_epoch = len(_train_dataloader)
    _max_train_steps = _param.epochs * _max_steps_per_epoch
    _warmup_steps = int(_max_train_steps * _param.warmup_ratio)

    _lr_scheduler = get_scheduler(
        name='linear',
        optimizer=_optimizer,
        num_warmup_steps=_warmup_steps,
        num_training_steps=_max_train_steps
    )






if __name__ == '__main__':
    model2train()
