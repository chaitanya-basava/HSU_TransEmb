import pandas as pd

import torch
from torch.utils.data import Dataset


class Task1Dataset(Dataset):
    def __init__(
        self,
        texts,
        labels,
        tokenizer,
        max_len=512,
        padding_type="max_length",
    ):

        self.max_len = max_len
        self.tokenizer = tokenizer
        self.padding_type = padding_type
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding=self.padding_type,
            return_attention_mask=True,
            return_tensors="pt",  # to get a torch.Tensor
            truncation=True,
        )

        return {
            "text": text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }
