import os
import sys
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn

from iteration import step
from utils.dataloader import get_testloader
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

print(f"Model with wted F1 = {best_weighted_f1} is loaded!!!")

test_loader = get_testloader(
    config["dataset"]["data_dir"],
    config["dataset"]["file_name"],
    config["model"]["model"],
    config["inference"]["batch_size"],
    config["hyperparameters"]["max_len"],
)


_model.eval()
y_preds = np.array([])
probs = np.array([])
with torch.set_grad_enabled(False):
    with tqdm(test_loader, desc="") as vepoch:
        for batch_idx, batch in enumerate(vepoch):
            ypred, probabilities = step(_model, batch, None, device, True)

            ypred = ypred.cpu().numpy()
            probabilities = probabilities.cpu().numpy()
            y_preds = np.hstack((y_preds, ypred))
            probs = np.hstack((probs, probabilities))


test_path = os.path.join(config["dataset"]["data_dir"], config["dataset"]["file_name"])
test = pd.read_csv(test_path, sep="\t", header=None)
if test.loc[0][0] == "Id" or test.loc[0][0] == "ID":
    test = test.drop([0])
test = test.reset_index(drop=True)
test.columns = ["id", "text"]
# test['label'] = np.where(y_preds==0, 'Not', 'Off').astype(str)
test["label"] = y_preds
test["probability"] = probs


test.to_csv(f"{model_dir}/_test.tsv", sep="\t", index=False)
print(f"Results stores in {model_dir}/_test.tsv")
