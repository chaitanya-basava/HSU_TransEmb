import os
import numpy as np
import pandas as pd

from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from dataset.dataset import TaskDataset
from dataset.clean_data import process_data


def get_testloader(
    data_path,
    file_name,
    model,
    mode,
    batch_size=64,
    max_length=256,
    padding_type="max_length",
):
    tokenizer = AutoTokenizer.from_pretrained(model)

    path = os.path.join(data_path, f"{file_name}{mode}.tsv")

    df = pd.read_csv(path, sep="\t")
    df = df[~df["category"].isin(["not-Tamil", "not-malayalam"])]

    print("Processing Data")
    df = process_data(df)
    print("Done Processing")

    dataset = TaskDataset(
        df,
        tokenizer=tokenizer,
        max_len=max_length,
        padding_type=padding_type,
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
    )
