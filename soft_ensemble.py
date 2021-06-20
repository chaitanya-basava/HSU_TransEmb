import os
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

with open("./config.yaml") as file:
    config = yaml.safe_load(file)

model_dir = os.path.join(
    config["model"]["model_loc"],
    config["dataset"]["file_name"],
)

_mode = config["ensemble"]["mode"]
notoff_probs = {}

for model_name in config["ensemble"]["models"]:
    csv_file = os.path.join(model_dir, model_name, f"_{_mode}.csv")
    df = pd.read_csv(csv_file)

    for _, row in df.iterrows():
        id = row["ids"]
        notoff_prob = row["probability"] if row["pred"] == 0 else 1 - row["probability"]

        if id in notoff_probs:
            notoff_probs[id]["prob"] += notoff_prob
        else:
            notoff_probs[id] = {"prob": notoff_prob, "true": row["true"]}

_ids, _probs = [], []
_labels, _true = [], []
for id, stats in notoff_probs.items():
    prob = stats["prob"] / 3
    true = stats["true"]

    label = 0 if prob > 0.5 else 1

    _ids.append(id)
    _true.append(true)
    _labels.append(label)
    _probs.append(prob if label == 0 else 1 - prob)

print(
    classification_report(
        np.array(_true),
        np.array(_labels),
        target_names=["OFF", "NOT"],
    )
)
print("wted-f1: {:.3f}".format(f1_score(_true, _labels, average="weighted")))

df = pd.DataFrame({"ids": _ids, "probability": _probs, "pred": _labels, "true": _true})
df.to_csv(f"{model_dir}/softensemble_{_mode}.csv", index=False)
