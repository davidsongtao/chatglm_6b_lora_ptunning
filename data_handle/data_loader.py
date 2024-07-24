from torch.utils.data import DataLoader
from transformers import default_data_collator, AutoTokenizer
from data_handle.data_preprocess import *
from glm_config import *

# 实例化配置文件和分词器
param = ParametersConfig()
tokenizer = AutoTokenizer.from_pretrained(param.pretrained_model,
                                          trust_remote_code=True,
                                          revision='main'
                                          )


def get_data():
    dataset = load_dataset(path='text',
                           data_files={'train': param.train_path, 'dev': param.dev_path})

    new_func = partial(
        convert_samples,
        tokenizer=tokenizer,
        max_source_seq_len=200,
        max_target_seq_len=100
    )

    dataset = dataset.map(new_func, batched=True)
    train_dataset = dataset['train']
    # print(train_dataset)
    dev_dataset = dataset['dev']
    # print(dev_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=param.batch_size,
        collate_fn=default_data_collator
    )

    dev_dataloader = DataLoader(
        dev_dataset,
        shuffle=True,
        batch_size=param.batch_size,
        collate_fn=default_data_collator
    )

    return train_dataloader, dev_dataloader


if __name__ == '__main__':
    train_dataloader, dev_dataloader = get_data()
