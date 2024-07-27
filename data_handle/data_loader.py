from functools import partial

from torch.utils.data import DataLoader
from transformers import default_data_collator, AutoTokenizer
from data_handle.data_handler import *
from glm_config import *


def get_dataloader(train_path, eval_path, tokenizer):

    dataset = load_dataset(path='text', data_files={'train': train_path, 'eval': eval_path})

    new_func = partial(
        convert_samples,
        tokenizer=tokenizer,
        max_context_len=300,
        max_target_len=200
    )

    dataset = dataset.map(new_func, batched=True)

    train_dataset = dataset['train']
    eval_dataset = dataset['eval']

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=default_data_collator,
        batch_size=param.batch_size,
        shuffle=True
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=default_data_collator,
        batch_size=param.batch_size,
        shuffle=True
    )

    return train_dataloader, eval_dataloader


if __name__ == '__main__':
    param = ParametersConfig()
    tokenizer = AutoTokenizer.from_pretrained(param.pretrained_model, trust_remote_code=True, revision='main')
    # dataset = load_dataset(path='text', data_files={'train': param.train_path, 'dev': param.dev_path})
    train_dataloader, eval_dataloader = get_dataloader(param.train_path, param.dev_path, tokenizer)
    print(f"train_dataloader --> {train_dataloader}")
    print(f"eval_dataloader --> {eval_dataloader}")
