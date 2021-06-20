import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

from transformers import AutoTokenizer
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset.dataset import TaskDataset, TestDataset
from dataset.clean_data import preprocess_text


def get_dataloader_task1(
    data_path,
    file_name,
    model,
    batch_size=64,
    max_length=256,
    padding_type="max_length",
):
    tokenizer = AutoTokenizer.from_pretrained(model)

    path = os.path.join(data_path, file_name + "train.tsv")

    df = pd.read_csv(path, sep="\t")
    df = df[~df["category"].isin(["not-Tamil"])]

    train, val = train_test_split(
        df, test_size=0.20, random_state=0, stratify=df["category"]
    )

    # DATA proportion
    grp = train.groupby(["category"])["id"].nunique()
    train_prop = {key: grp[key] for key in list(grp.keys())}
    class_wts = np.array([1.0 / train_prop[c] for c in ["NOT", "OFF"]])

    print(class_wts)

    # class_wts = {c: train_prop[c] / len(train) for c in ["NOT", "OFF"]}
    # example_wts = [1./class_wts[c] for c in train.category]
    # sampler = WeightedRandomSampler(example_wts, len(train))

    train_dataset = TaskDataset(
        train,
        tokenizer=tokenizer,
        max_len=max_length,
        padding_type=padding_type,
    )

    val_dataset = TaskDataset(
        val,
        tokenizer=tokenizer,
        max_len=max_length,
        padding_type=padding_type,
    )

    return (
        DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
        ),
        DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
        ),
        torch.tensor(class_wts, dtype=torch.float),
    )


def get_dataloader_task2(
    data_path,
    file_name,
    model,
    batch_size=64,
    max_length=256,
    padding_type="max_length",
):
    tokenizer = AutoTokenizer.from_pretrained(model)

    train_path = os.path.join(data_path, file_name + "train.tsv")
    train = pd.read_csv(train_path, sep="\t")

    val_path = os.path.join(data_path, file_name + "dev.tsv")
    val = pd.read_csv(val_path, sep="\t")

    # DATA proportion
    grp = train.groupby(["category"])["id"].nunique()
    train_prop = {key: grp[key] for key in list(grp.keys())}
    class_wts = np.array([1.0 - train_prop[c] / len(train) for c in list(grp.keys())])

    print(class_wts)

    # class_wts = {c: train_prop[c] / len(train) for c in ["NOT", "OFF"]}
    # example_wts = [1./class_wts[c] for c in train.category]
    # sampler = WeightedRandomSampler(example_wts, len(train))

    train_dataset = TaskDataset(
        train,
        tokenizer=tokenizer,
        max_len=max_length,
        padding_type=padding_type,
    )

    val_dataset = TaskDataset(
        val,
        tokenizer=tokenizer,
        max_len=max_length,
        padding_type=padding_type,
    )

    return (
        DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
        ),
        DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
        ),
        torch.tensor(class_wts, dtype=torch.float),
    )

def get_testloader(
    data_path,
    file_name,
    model,
    batch_size=64,
    max_length=256,
    padding_type="max_length",
):
    tokenizer = AutoTokenizer.from_pretrained(model)

    test_path = os.path.join(data_path, file_name)
    test = pd.read_csv(test_path, sep="\t", header=None)
    if(test.loc[0][0] == 'Id' or test.loc[0][0] == 'ID' ):
        test = test.drop([0])
    test = test.reset_index(drop=True)
    test.columns = ['id', 'text']

    print("Processing Data")
    test["cleaned_text"] = test["text"].map(lambda x: preprocess_text(x))
    print("Done Processing")

    test_dataset = TestDataset(
        test,
        tokenizer=tokenizer,
        max_len=max_length,
        padding_type=padding_type,
    )


    return (
        DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=True,
        ),
    )
