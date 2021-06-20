import os
import yaml
import pandas as pd

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
        id = row["id"]
        notoff_prob = (
            row["probability"] if row["label"] == 0 else 1 - row["probability"]
        )

        if id in notoff_probs:
            notoff_probs[id]["prob"] += notoff_prob
        else:
            notoff_probs[id] = {"prob": notoff_prob, "text": row["text"]}

_ids, _probs = [], []
_labels, _text = [], []
for id, stats in notoff_probs.items():
    prob = stats["prob"] / 3
    text = stats["text"]

    label = 0 if prob > 0.5 else 1

    _ids.append(id)
    _text.append(text)
    _labels.append(label)
    _probs.append(prob if label == 0 else 1 - prob)

df = pd.DataFrame({"id": _ids, "text": _text, "pred": _labels})
df.to_csv(
    f"{model_dir}/{config['dataset']['file_name']}_{_mode}_preds.tsv",
    sep="\t",
    index=False,
)
