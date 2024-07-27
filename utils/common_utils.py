"""
Description: 
    
-*- Encoding: UTF-8 -*-
@File     ：common_utils.py
@Author   ：King Songtao
@Time     ：2024/7/27 下午10:26
@Contact  ：king.songtao@gmail.com
"""
import torch
from torch import nn


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)
