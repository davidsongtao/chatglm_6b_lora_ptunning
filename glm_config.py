import torch.cuda


class ParametersConfig:
    def __init__(self):
        # 定义是否使用GPU
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.pretrained_model = r'D:\Projects\chatglm_lora_ptuning\models\ChatGLM-6B'
        # 定义训练数据的路径
        self.train_path = r'D:\Projects\chatglm_lora_ptuning\data\mixed_train_dataset.jsonl'
        # 定义验证集数据的路径
        self.eval_path = r'D:\Projects\chatglm_lora_ptuning\data\mixed_dev_dataset.jsonl'
        # 是否使用lora方法微调
        self.use_lora = True
        # 是否使用P-Tunning方法微调
        self.use_ptuning = False
        # 秩 == 8
        self.lora_rank = 8
        # 一个批次多少个样本
        self.batch_size = 1
        # 训练几轮
        self.epochs = 2
        # 学习率
        self.learning_rate = 3e-5
        # 权重衰减系数
        self.weight_decay = 0
        # 学习率预热比例
        self.warmup_ratio = 0.06
        # context文本的输入长度限制
        self.max_source_sq_len = 300
        # target文本长度限制
        self.max_target_sq_len = 200
        # 每隔多少步打印日志
        self.logging_steps = 10
        # 每隔多少步保存
        self.save_freq = 200
        # 如果使用了P-Tunning,要定义伪tokens的长度
        self.pre_seq_len = 128
        # 默认为false,即P-Tunning,如果为True,即为P-Tunning-v2
        self.prefix_projection = False
        # 保存模型的路径
        self.save_dir = r'D:\lora\models\save_model'