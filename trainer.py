import os
import sys
import yaml
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup, AdamW

from transformer.model import TransformerClassifier
from utils.dataloader import get_dataloader_task1


with open("./config.yaml") as file:
    config = yaml.safe_load(file)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_dir = config["model"]["model_loc"]
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_dir = os.path.join(model_dir, config['model']['model'].split('/')[-1])
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

def step(model, batch):
    targets = batch["label"].to(device)

    outputs = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
    )

    _, preds = torch.max(outputs, dim=1)

    return {
        "loss": nn.functional.cross_entropy(outputs, targets),
        "accuracy": (preds == targets).float().mean(),
    }


def configure_optimizers(model, dataloader):
    optim = AdamW(
        model.parameters(),
        lr=float(config["hyperparameters"]["lr"]),
        correct_bias=False,
    )
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=0,
        num_training_steps=len(dataloader) * config["hyperparameters"]["epochs"],
    )
    return optim, scheduler

_model = TransformerClassifier(config["model"]["model"]).to(device)

train_dataloader, val_dataloader = get_dataloader_task1(
    config["dataset"]["data_dir"],
    config["dataset"]["file_name"],
    config["model"]["model"],
    config["hyperparameters"]["batch_size"],
)

optimizer, scheduler = configure_optimizers(_model, train_dataloader)

if config["model"]["ckpt"]:
    checkpoint = torch.load(
        os.path.join(model_dir, config["model"]["ckpt"]), map_location=device
    )

    _model.load_state_dict(checkpoint[config["model"]["model"]])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    best_val_acc = checkpoint["val_acc"]
    start_epoch = checkpoint["epoch"]
else:
    start_epoch, best_val_acc = 1, 0.0

total_epochs = config["hyperparameters"]["epochs"] + start_epoch

for epoch in range(start_epoch, total_epochs):
    train_loss, train_acc = [], []
    val_loss, val_acc = [], []

    #### TRAIN STEP ####
    _model.train()
    with tqdm(
        train_dataloader, desc="train-{}/{}".format(epoch, total_epochs - 1)
    ) as tepoch:
        tepoch.set_postfix(loss=0.0, acc=0.0)
        for batch_idx, batch in enumerate(tepoch):
            details = step(_model, batch)

            optimizer.zero_grad()
            details["loss"].backward()
            optimizer.step()
            scheduler.step()

            train_loss.append(details["loss"].item())
            train_acc.append(details["accuracy"].item())

            tepoch.set_postfix(
                loss=details["loss"].item(), acc=np.array(train_acc).mean()
            )

    #### VAL STEP ####
    _model.eval()
    with torch.set_grad_enabled(False):
        with tqdm(
            val_dataloader, desc="val-{}/{}".format(epoch, total_epochs - 1)
        ) as vepoch:
            vepoch.set_postfix(loss=0.0, acc=0.0)
            for batch_idx, batch in enumerate(vepoch):
                details = step(_model, batch)

                val_loss.append(details["loss"].item())
                val_acc.append(details["accuracy"].item())

                vepoch.set_postfix(
                    loss=details["loss"].item(), acc=np.array(val_acc).mean()
                )

    avg_val_acc = np.array(val_acc).mean()
    if best_val_acc <= avg_val_acc:
        best_val_acc = avg_val_acc
        torch.save(
            {
                config["model"]["model"]: _model.state_dict(),
                "scheduler": scheduler.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": start_epoch + epoch,
                "val_acc": best_val_acc,
            },
            os.path.join(model_dir, f"{epoch}_{int(avg_val_acc*100.)}.pkl"),
        )

    print(
        "Epoch {} - train loss: {}, train acc: {}".format(
            epoch,
            np.array(train_loss).mean(),
            np.array(train_acc).mean() * 100.0,
            np.array(val_loss).mean(),
            avg_val_acc * 100.0,
        )
    )
