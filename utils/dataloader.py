import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset.Task1Dataset import Task1Dataset


def get_dataloader_task1(
    data_path,
    file_name,
    model,
    batch_size=64,
    mode="train",
    max_length=512,
    padding_type="max_length",
):
    tokenizer = AutoTokenizer.from_pretrained(model)

    path = os.path.join(data_path, file_name + mode + ".tsv")
    df = pd.read_csv(path, sep="\t")

    texts = df.cleaned_text.to_numpy()
    labels = np.array(df.category.to_numpy() == "OFF", dtype=np.int)

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.33, random_state=0
    )

    train_dataset = Task1Dataset(
        X_train,
        y_train,
        tokenizer=tokenizer,
        max_len=max_length,
        padding_type=padding_type,
    )

    val_dataset = Task1Dataset(
        X_val,
        y_val,
        tokenizer=tokenizer,
        max_len=max_length,
        padding_type=padding_type,
    )

    return DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    ), DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
