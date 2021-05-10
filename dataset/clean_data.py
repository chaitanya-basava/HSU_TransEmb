import os
import re
import pandas as pd

import argparse

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
        text = text.replace(chr(i), " ")
    text = text.replace("\n", ".")
    text = text.replace("&amp", " ")
    text = text.replace("-", "")
    text = text.replace(";", "")
    text = text.replace("@", "")

    emoji = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji.sub(r"", text)

    return text


def process_csv(_type):
    path = os.path.join(args.data, args.file + _type + ".tsv")
    df = pd.read_csv(path, sep="\t")

    df["cleaned_text"] = df["text"].map(lambda x: preprocess_text(x))

    df.to_csv(path, sep="\t", index=False)


process_csv("train")
# process_csv("val")
# process_csv("test")
