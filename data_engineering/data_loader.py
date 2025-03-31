# data_loader.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer
from torch.utils.data import Dataset
import torch

from data_engineering.config import EMOTION_LABELS, CSV_URLS, LOCAL_CSVS


def download_and_load_data():
    for url, filename in zip(CSV_URLS, LOCAL_CSVS):
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            pd.read_csv(url).to_csv(filename, index=False)
        else:
            print(f"{filename} already exists.")

    df_list = [pd.read_csv(f) for f in LOCAL_CSVS]
    df = pd.concat(df_list).reset_index(drop=True)
    return df


def load_and_prepare_data(sample_frac=0.2):
    df = download_and_load_data()
    df = df[df['text'].notnull()]
    df = df.sample(frac=sample_frac, random_state=42)
    df[EMOTION_LABELS] = df[EMOTION_LABELS].fillna(0)

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['text'], df[EMOTION_LABELS], test_size=0.2, random_state=42
    )
    return train_texts, test_texts, train_labels, test_labels


def get_tokenizer():
    return DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


class GoEmotionsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels.values, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}, self.labels[idx]

