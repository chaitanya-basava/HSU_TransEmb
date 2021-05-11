import os
import re
import demoji
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
demoji.download_codes() 


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

    # replace emoji's with text decription
    # emoji_dict = demoji.findall(text)
    # if len(emoji_dict): 
    #     for emoji, emoji_text in emoji_dict.items():
    #         text = text.replace(emoji, ' '+emoji_text+' ')
    #         text = ' '.join(text.split())

    # remove emoji
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
