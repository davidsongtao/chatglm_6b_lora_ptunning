# ChatGLM 6B with LoRA P-Tuning 🚀
通过使用LoRA对ChatGLM-6B开源大模型进行微调，从而实现利用ChatGLM大模型进行复合任务处理。本项目中主要处理两个任务：新媒体行业评论只能分类与信息抽取

<!-- PROJECT SHIELDS -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]

## 项目背景介绍

本项目使用基座模型：ChatGLM 6B

### 硬件GPU要求

| **Quantization Level** | **GPU Memory** |
|------------------------|----------------|
| FP16（no quantization）  | 13 GB        |

### LoRA 原理介绍

LoRA技术冻结预训练模型的权重，并在每个Transformer块中注入可训练层（称为秩分解矩阵），即在模型的Linear层的旁边增加一个“旁支”A和B。其中，A将数据从d维降到r维，这个r是LoRA的秩，是一个重要的超参数；B将数据从r维升到d维，B部分的参数初始为0。模型训练结束后，需要将A+B部分的参数与原大模型的参数合并在一起使用。

### 数据介绍

数据格式：字典样式；context内容代表：原始输入文本（prompt）；target指向：目标文本

训练数据集共计包含：902条样本

验证数据集共计包含：122条样本

## 许可选项

本项目中涉及ChatGLM-6B 模型权重的使用受到 [模型许可](MODEL_LICENSE) 的约束。








<!-- links -->
[contributors-shield]: https://img.shields.io/github/contributors/shaojintian/Best_README_template.svg?style=flat-square
[contributors-url]: https://github.com/davidsongtao/chatglm_6b_lora_ptunning/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/shaojintian/Best_README_template.svg?style=flat-square
[forks-url]: https://github.com/davidsongtao/chatglm_6b_lora_ptunning/network/members
[stars-shield]: https://img.shields.io/github/stars/shaojintian/Best_README_template.svg?style=flat-square
[stars-url]: https://github.com/davidsongtao/chatglm_6b_lora_ptunning/stargazers
