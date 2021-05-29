import pandas as pd

import torch
from torch.utils.data import Dataset


class Task1Dataset(Dataset):
    def __init__(
        self,
        df,
        tokenizer,
        max_len=512,
        padding_type="max_length",
    ):

        self.max_len = max_len
        self.tokenizer = tokenizer
        self.padding_type = padding_type
        self.texts = df.cleaned_text.to_numpy()
        self.labels = df.category.to_numpy()

        self.label_dict = {
            "NOT": 0,
            "OFF": 1,
            "not-Tamil": 2,
        }

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.label_dict[self.labels[item].upper()]

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
