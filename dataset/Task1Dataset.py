import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset


class Task1Dataset(Dataset):
    def __init__(
        self,
        path,
        file,
        _type,
        tokenizer,
        max_len=512,
        padding_type="max_length",
    ):

        self.max_len = max_len
        self.tokenizer = tokenizer
        self.padding_type = padding_type
        self.path = os.path.join(path, file + _type + ".tsv")

        df = pd.read_csv(self.path, sep='\t')
        self.texts = df.cleaned_text.to_numpy()
        self.labels = np.array(df.category.to_numpy() == "OFF", dtype=np.int)

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
