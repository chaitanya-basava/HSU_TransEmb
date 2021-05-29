import os
import yaml
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import get_linear_schedule_with_warmup, AdamW

from transformer.model import TransformerClassifier
from utils.dataloader import get_dataloader_task1, get_dataloader_task2
from utils.load_checkpoint import load_checkpoint


with open("./config.yaml") as file:
    config = yaml.safe_load(file)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_dir = config["model"]["model_loc"]
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_dir = os.path.join(model_dir, config["dataset"]["file_name"])
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_dir = os.path.join(model_dir, config["model"]["model"].split("/")[-1])
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


def step(model, batch, criterion):
    targets = batch["label"].to(device)

    outputs = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
    )

    _, preds = torch.max(outputs, dim=1)

    return (
        {
            "loss": criterion(outputs, targets),
            "accuracy": (preds == targets).float().mean(),
        },
        preds,
        targets,
    )


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


_model = TransformerClassifier(
    config["model"]["model"],
    hidden_states=config["hyperparameters"]["hidden_layers"],
    dropout=config["hyperparameters"]["dropout"],
).to(device)

train_dataloader, val_dataloader, class_wt = get_dataloader_task2(
    config["dataset"]["data_dir"],
    config["dataset"]["file_name"],
    config["model"]["model"],
    config["hyperparameters"]["batch_size"],
    max_len=config["hyperparameters"]["max_len"],
)

criterion = nn.CrossEntropyLoss(
    weight=class_wt.to(device) if config["hyperparameters"]["use_weights"] else None
)
optimizer, scheduler = configure_optimizers(_model, train_dataloader)

_model, optimizer, scheduler, best_weighted_f1, start_epoch = load_checkpoint(
    config["model"]["ckpt"],
    config["model"]["model"],
    model_dir,
    device,
    _model,
    optimizer,
    scheduler,
)

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
            details, _, _ = step(_model, batch, criterion)

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
    y_preds, y_test = np.array([]), np.array([])
    with torch.set_grad_enabled(False):
        with tqdm(
            val_dataloader, desc="val-{}/{}".format(epoch, total_epochs - 1)
        ) as vepoch:
            vepoch.set_postfix(loss=0.0, acc=0.0)
            for batch_idx, batch in enumerate(vepoch):
                details, ypred, ytrue = step(_model, batch, criterion)

                y_preds = np.hstack((y_preds, ypred.cpu().numpy()))
                y_test = np.hstack((y_test, ytrue.to("cpu").numpy()))

                val_loss.append(details["loss"].item())
                val_acc.append(details["accuracy"].item())

                vepoch.set_postfix(
                    loss=details["loss"].item(), acc=np.array(val_acc).mean()
                )

    weighted_f1 = f1_score(y_test, y_preds, average="weighted")

    # avg_val_acc = np.array(val_acc).mean()
    # if best_val_acc <= avg_val_acc:
    #     best_val_acc = avg_val_acc
    #     torch.save(
    #         {
    #             config["model"]["model"]: _model.state_dict(),
    #             "scheduler": scheduler.state_dict(),
    #             "optimizer": optimizer.state_dict(),
    #             "epoch": start_epoch + epoch,
    #             "val_acc": best_val_acc,
    #         },
    #         os.path.join(model_dir, f"{epoch}_{int(avg_val_acc*100.)}.pkl"),
    #     )

    if best_weighted_f1 <= weighted_f1:
        best_weighted_f1 = weighted_f1
        torch.save(
            {
                config["model"]["model"]: _model.state_dict(),
                "scheduler": scheduler.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": start_epoch + epoch,
                "weighted_f1": weighted_f1,
            },
            os.path.join(model_dir, f"{epoch}_{int(weighted_f1*100.)}.pkl"),
        )

    print(
        "Epoch {:.3f} - train loss: {:.3f}, train acc: {:.3f}, val loss: {:.3f}, val acc: {:.3f}, wted-f1: {:.3f}".format(
            epoch,
            np.array(train_loss).mean(),
            np.array(train_acc).mean() * 100.0,
            np.array(val_loss).mean(),
            np.array(val_acc).mean() * 100.0,
            weighted_f1 * 100.0,
        )
    )
