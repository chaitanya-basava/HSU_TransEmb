import os
import yaml
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report

import torch
import torch.nn as nn

from iteration import step, configure_optimizers
from utils.load_checkpoint import load_checkpoint
from transformer.model import TransformerClassifier
from utils.dataloader import get_dataloader_task1, get_dataloader_task2


with open("./config.yaml") as file:
    config = yaml.safe_load(file)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
main_checkpoint_dir = os.path.join(
    config["model"]["model_loc"],
    config["dataset"]["file_name"],
)
