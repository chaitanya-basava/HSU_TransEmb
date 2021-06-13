import os
import sys
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report

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
    config["inference"]["batch_size"],
    config["hyperparameters"]["max_len"],
)

_ids, _probs, _labels = [], [], []
test_loss, test_acc = [], []

_model.eval()
y_preds, y_test = np.array([]), np.array([])
with torch.set_grad_enabled(False):
    with tqdm(test_loader, desc="") as vepoch:
        vepoch.set_postfix(loss=0.0, acc=0.0)
        for batch_idx, batch in enumerate(vepoch):
            details, ypred, ytrue, probabilities, representations = step(
                _model, batch, criterion, device
            )

            ypred = ypred.cpu().numpy()

            if config["inference"]["save_representations"]:
                ids = batch["id"]
                representations = representations.cpu().numpy()
                probabilities = probabilities.cpu().numpy()

                for i, j, k, l in zip(ids, representations, probabilities, ypred):
                    _ids.append(i)
                    _probs.append(k)
                    _labels.append(l)
                    np.save(f"{rep_dir}/{i}.npy", j)

            y_preds = np.hstack((y_preds, ypred))
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

df = pd.DataFrame({"IDS": _ids, "probability": _probs, "label": _labels})
df.to_csv(f"{model_dir}/_{config['inference']['mode']}.csv", index=False)
print(f"Results stores in {model_dir}/_{config['inference']['mode']}.csv")
