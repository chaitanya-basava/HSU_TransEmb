import os, sys
import numpy as np

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset.Task1Dataset import Task1Dataset


def get_dataloader(
    data_path,
    file_name,
    model,
    batch_size=64,
    mode="train",
    max_length=512,
    padding_type="max_length",
):
    tokenizer = AutoTokenizer.from_pretrained(model)

    dataset = Task1Dataset(
        path=data_path,
        file=file_name,
        _type=mode,
        tokenizer=tokenizer,
        max_len=max_length,
        padding_type=padding_type,
    )

    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=(mode == "train"))
