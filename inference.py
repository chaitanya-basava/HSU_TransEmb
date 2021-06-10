import os
import sys
import yaml
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report

import torch
import torch.nn as nn

from iteration import step
from utils.infereloader import get_testloader
from utils.load_checkpoint import load_checkpoint
from transformer.model import TransformerClassifier

with open("./config.yaml") as file:
    config = yaml.safe_load(file)

if not config["model"]["ckpt"]:
    print("Checkpoint is needed!!!")
    sys.exit(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_dir = os.path.join(
    config["model"]["model_loc"],
    config["dataset"]["file_name"],
    config["model"]["model"].split("/")[-1],
)

if config["inference"]["save_representations"]:
    rep_dir = os.path.join(model_dir, "representations")
    if not os.path.exists(rep_dir):
        os.makedirs(rep_dir)

_model = TransformerClassifier(
    config["model"]["model"],
    hidden_states=config["hyperparameters"]["hidden_layers"],
    dropout=config["hyperparameters"]["dropout"],
).to(device)

print(config["model"]["ckpt"])

_model, _, _, best_weighted_f1, _ = load_checkpoint(
    config["model"]["ckpt"],
    config["model"]["model"],
    model_dir,
    device,
    _model,
    None,
    None,
)

criterion = nn.CrossEntropyLoss()

print(f"Model with wted F1 = {best_weighted_f1} is loaded!!!")

test_loader = get_testloader(
    config["dataset"]["data_dir"],
    config["dataset"]["file_name"],
    config["model"]["model"],
    config["inference"]["mode"],
    config["hyperparameters"]["batch_size"],
    config["hyperparameters"]["max_len"],
)

test_loss, test_acc = [], []

_model.eval()
y_preds, y_test = np.array([]), np.array([])
with torch.set_grad_enabled(False):
    with tqdm(test_loader, desc="") as vepoch:
        vepoch.set_postfix(loss=0.0, acc=0.0)
        for batch_idx, batch in enumerate(vepoch):
            details, ypred, ytrue, representations = step(
                _model, batch, criterion, device
            )

            if config["inference"]["save_representations"]:
                ids = batch["id"]
                representations = representations.cpu().numpy()

                for i, j in zip(ids, representations):
                    np.save(f"{rep_dir}/{i}.npy", j)

            y_preds = np.hstack((y_preds, ypred.cpu().numpy()))
            y_test = np.hstack((y_test, ytrue.to("cpu").numpy()))

            test_loss.append(details["loss"].item())
            test_acc.append(details["accuracy"].item())

            vepoch.set_postfix(
                loss=details["loss"].item(), acc=np.array(test_acc).mean()
            )

print(
    classification_report(
        y_test,
        y_preds,
        target_names=["OFF", "NOT"],
    )
)

print(f"{config['inference']['mode']} data processed")
