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

# if not config["model"]["ckpt"]:
#     print("Checkpoint is needed!!!")
#     sys.exit(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# model_dir = os.path.join(
#     config["model"]["model_loc"],
#     config["dataset"]["file_name"],
#     config["model"]["model"].split("/")[-1],
# )

# _model = TransformerClassifier(
#     config["model"]["model"],
#     hidden_states=config["hyperparameters"]["hidden_layers"],
#     dropout=config["hyperparameters"]["dropout"],
# ).to(device)

# _model, _, _, best_weighted_f1, _ = load_checkpoint(
#     config["model"]["ckpt"],
#     config["model"]["model"],
#     model_dir,
#     device,
#     _model,
#     None,
#     None,
# )

# criterion = nn.CrossEntropyLoss()

# print(f"Model with wted F1 - {best_weighted_f1} is loaded!!!")

test_loader = get_testloader(
    config["dataset"]["data_dir"],
    config["dataset"]["file_name"],
    config["model"]["model"],
    config["inference"]["mode"],
    config["hyperparameters"]["batch_size"],
    config["hyperparameters"]["max_len"],
)

test_loss, test_acc = [], []
rep_data = {}

# _model.eval()
y_preds, y_test = np.array([]), np.array([])
with torch.set_grad_enabled(False):
    with tqdm(test_loader, desc="") as vepoch:
        for batch_idx, batch in enumerate(vepoch):
            print(batch)
            sys.exit(0)
            # details, ypred, ytrue, representations = step(_model, batch, criterion, device)

            # if config["inference"]["save_representations"]:
            #     text = batch["text"]
            #     representations = representations.cpu().numpy()

            #     for i, j in zip(batch, representations):
            #         rep_data[i["text"]] = representations

            # y_preds = np.hstack((y_preds, ypred.cpu().numpy()))
            # y_test = np.hstack((y_test, ytrue.to("cpu").numpy()))

            # test_loss.append(details["loss"].item())
            # test_acc.append(details["accuracy"].item())

            # vepoch.set_postfix(
            #     loss=details["loss"].item(), acc=np.array(test_acc).mean()
            # )

# print(
#     classification_report(
#         y_test,
#         y_preds,
#         target_names=["OFF", "NOT"],
#     )
# )

# if config["inference"]["save_representations"]:
#     np.save(f"{model_dir}/representations.npy", rep_data)
