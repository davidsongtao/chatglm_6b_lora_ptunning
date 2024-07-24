"""
直观地观看分析数据集中数据样本长什么样子
"""

# 分类任务数据样本示例
classification_sample = {
    "context": "Instruction: 你现在是一个很厉害的阅读理解器，严格按照人类指令进行回答。"
               "Input: 下面句子可能是一条关于什么的评论，用列表形式回答："

               "我在尼斯住了5天，价格不是很贵，服务比较细腻，早餐也比较丰富，感觉还是不错。"
               "Answer: ",
    "target": "['酒店']"
}

# 信息抽取任务数据样本示例
extraction_sample = {
    "context": "Instruction: 你现在是一个很厉害的阅读理解器，严格按照人类指令进行回答。"
               "Input: 抽出下面句子中的关系信息，并输出为json："
               ""
               "《青楼大老板》是在17k小说网连载的架空历史小说，作者是好一朵美丽的茉莉花。"
               "Answer: ",
    "target": "```json"
              "["
              "     {"
              "         'predicate': '连载网站', "
              "         'object_type': '网站', "
              "         'subject_type': '网络小说', "
              "         'object': '17k小说网', "
              "         'subject': '青楼大老板'"
              "     }, "
              "     {"
              "         'predicate': '作者', "
              "         'object_type': '人物', "
              "         'subject_type': '图书作品', "
              "         'object': '好一朵美丽的茉莉花', "
              "         'subject': '青楼大老板'"
              "     }"
              "]"
              "```"
}
