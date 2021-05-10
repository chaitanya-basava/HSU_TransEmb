import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="")

parser.add_argument(
    "--data",
    dest="data",
    default="./data",
)


args = parser.parse_args()
train_data = pd.DataFrame()
val_data = pd.DataFrame()
test_data = pd.DataFrame()


def process_tweet(x):
    x = x.replace("...", ".")
    x = x.replace(".", " . ")
    return x


list_dir = [
    name
    for name in os.listdir(args.data)
    if os.path.isdir(os.path.join(args.data, name))
]
for j in list_dir:
    print(j, "\n-------")
    for i in os.listdir(os.path.join(args.data, j)):
        print(i)
        path = os.path.join(args.data, j, i)
        df = pd.read_csv(path)

        df["tweet"] = df["cleaned_tweet"].apply(process_tweet)
        df["lang"] = j
        df = df.drop(["cleaned_tweet"], axis=1)
        if i == "test.csv":
            test_data = test_data.append(df, ignore_index=True)
        if i == "val.csv":
            val_data = val_data.append(df, ignore_index=True)
        if i == "train.csv":
            train_data = train_data.append(df, ignore_index=True)


train_data.to_csv(os.path.join(args.data, "train.csv"), sep=",", index=False)

val_data.to_csv(os.path.join(args.data, "val.csv"), sep=",", index=False)

test_data.to_csv(os.path.join(args.data, "test.csv"), sep=",", index=False)
