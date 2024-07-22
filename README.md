# ChatGLM 6B with LoRA P-Tuning 🚀
通过使用LoRA对ChatGLM-6B开源大模型进行微调，从而实现利用ChatGLM大模型进行复合任务处理。本项目中主要处理两个任务：新媒体行业评论只能分类与信息抽取

<!-- PROJECT SHIELDS -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]

## 项目背景介绍

### 文本分类概念
文本分类是指将一段或多段文本按照其内容或主题特征划分到不同的类别或标签中的过程。在实际工作中文本分类应用非常广泛，比如：新闻分类、简历分类、邮件分类、办公文档分类、区域分类等诸多方面，还能够实现文本过滤，从大量文本中快速识别和过滤出符合特殊要求的信息。

### 信息抽取概念
信息抽取，是从无结构或半结构化的自然文本中识别出实体、关系、事件等事实描述，以结构化的形式存储和利用的技术。以“小明和小秦是很好的朋友，他们都属于云南人，小明住在大理，小秦住在丽江。”为例，可以得到如：<小明，朋友，小秦>和<小秦，住在，丽江>和<小明，住在，大理>等三元组信息。

### 项目背景

随着互联网技术的快速发展，新媒体行业已经成为信息传播的主要平台之一。在这个信息爆炸的时代，人们通过社交媒体、新闻客户端、博客等多种形式获取信息。然而，随着信息量的不断增加，如何高效地管理和利用这些信息成为了亟待解决的问题。本项目基于部分“新媒体行业”数据为背景，通过文本评论的分类和信息抽取，帮助新媒体行业从海量的信息中快速准确地获取有用的信息，并进行合理的分类和管理。这不仅有助于新媒体平台提升用户体验，还能够为信息生产者提供更精准的数据分析和决策支持。

### 技术选型

基于ChatGLM-6B模型+LoRA微调方法，实现文本分类及信息抽取的联合任务的开发

### 硬件GPU要求

| **Quantization Level** | **GPU Memory** |
|------------------------|----------------|
| FP16（no quantization）  | 13 GB        |

### 环境准备

| **依赖包** | **版本要求** |
|---------------|------------------|
| protobuf      | >=3.19.5,<3.20.1 |
| transformers  | >=4.27.1         |
| icetk         | n/a              |
| cpm_kernels   | n/a              |
| streamlit     | ==1.17.0         |
| matplotlib    | n/a              |
| datasets      | >==2.10.1        |
| accelerate    | ==0.17.1         |
| packaging     | >=20.0           |
| psutil        | n/a              |
| pyyaml        | n/a              |
| peft          | n/a              |

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
