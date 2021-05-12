import os
import re
import argparse
import pandas as pd

from processing_util import emoticon_list, emoji

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--file",
    dest="file",
    default="tamil_offensive_",
)
parser.add_argument(
    "--data",
    dest="data",
    default="./data",
)
args = parser.parse_args()


def preprocess_text(text):

    text = text.lower()
    text = text.strip()
    text = re.sub(r"http\S+", "", text, flags=re.MULTILINE)  # urls
    text = re.sub(r"www\S+", "", text, flags=re.MULTILINE)  # urls
    # text = re.sub(r"@\S+", "", text, flags=re.MULTILINE)  # mentions
    text = re.sub(r"&\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\.{2,}", ".", text, flags=re.MULTILINE)  # cont .'s
    text = re.sub(r"&", " and ", text)

    for i in [8221, 35, 36, 94, 8217]:
        text = text.replace(chr(i), "")
    for i in emoticon_list:
        text = text.replace(i, "")
    text = text.replace("\n", ".")
    text = text.replace("&amp", " ")
    text = text.replace("-", "")
    text = text.replace(";", "")
    text = text.replace("@", "")

    # remove emoji
    text = emoji.sub(r"", text)

    return text


def process_csv(_type, _csv="tsv", names=None):
    path = os.path.join(args.data, args.file + _type + "." + _csv)

    df = pd.read_csv(path, sep="\t", names=names)
    df = df[~df["category"].isin(["not-Tamil", "not-malayalam"])]
    # df = df.dropna()

    print(df.head())
    
    df["cleaned_text"] = df["text"].map(lambda x: preprocess_text(x))
    df["category"] = df["category"].map(
        lambda x: "NOT" if (x == "Not_offensive") else "OFF"
    )

    # df["cleaned_text"] = df["Tweets"].map(lambda x: preprocess_text(x))
    # df = df.rename(columns = {'Labels': 'category'}, inplace = False)
    # df = df.rename(columns = {'ID': 'id'}, inplace = False)

    # df.to_csv(path.replace('Tamil/', ''), sep="\t", index=False)
    df.to_csv(path.replace('Malayalam/', ''), sep="\t", index=False)


# process_csv("train", "csv", ["text", "category", "category2"])
# process_csv("dev", "csv", ["text", "category", "category2"])

# process_csv("train")
# process_csv("train", "xlsx", ["id", "text", "category"])
# process_csv("dev", ["id", "text", "category"])
# process_csv("test")
