import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import torch
from torch.utils.data import Dataset, DataLoader
#from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW, get_scheduler
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, get_scheduler
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import json

# ========= Step 1: Load and Save Dataset Locally ============
urls = [
    "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv",
    "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv",
    "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv"
]
local_files = ["goemotions_1.csv", "goemotions_2.csv", "goemotions_3.csv"]

for url, filename in zip(urls, local_files):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        pd.read_csv(url).to_csv(filename, index=False)
    else:
        print(f"{filename} already exists.")

df_list = [pd.read_csv(f) for f in local_files]
df = pd.concat(df_list).reset_index(drop=True)

# ========= Step 2: Prepare Multi-label Classification ============
emotion_labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
    'remorse', 'sadness', 'surprise', 'neutral'
]

df = df[df['text'].notnull()]
df = df.sample(frac=0.2, random_state=42)
df[emotion_labels] = df[emotion_labels].fillna(0)

# ========= Step 3: Split and Tokenize ========================
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'], df[emotion_labels], test_size=0.2, random_state=42)

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(texts):
    return tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt", max_length=128)

train_encodings = tokenize(train_texts)
test_encodings = tokenize(test_texts)

# ========= Step 4: Dataset ===================================
class GoEmotionsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels.values, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}, self.labels[idx]

train_dataset = GoEmotionsDataset(train_encodings, train_labels)
test_dataset = GoEmotionsDataset(test_encodings, test_labels)

# ========= Step 5: Load Model ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=len(emotion_labels), problem_type="multi_label_classification"
)
model.to(device)

# ========= Step 6: Training ==================================
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0,
                             num_training_steps=len(train_loader) * num_epochs)
loss_fn = torch.nn.BCEWithLogitsLoss()

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        inputs, labels = batch
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        outputs = model(**inputs)
        loss = loss_fn(outputs.logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} loss: {total_loss / len(train_loader):.4f}")

torch.save(model.state_dict(), "distilbert_goemotions_multilabel.pt")
print("Model saved to 'distilbert_goemotions_multilabel.pt'.")

# ========= Step 7: Evaluation ================================
model.eval()
all_probs = []
true_labels = []

with torch.no_grad():
    for batch in test_loader:
        inputs, labels = batch
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()
        all_probs.append(probs)
        true_labels.append(labels.numpy())

all_probs = np.vstack(all_probs)
true_labels = np.vstack(true_labels)

threshold = 0.5
preds = (all_probs >= threshold).astype(int)

print("\nClassification Report (threshold = 0.5):")
print(classification_report(true_labels, preds, target_names=emotion_labels, zero_division=0))

# ========= Step 8: Optional - Macro F1 =========================
macro_f1 = f1_score(true_labels, preds, average="macro", zero_division=0)
print(f"\nMacro F1 Score: {macro_f1:.4f}")

# === Step 9:Save Precision/Recall/F1 for all label ===
from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1s, support = precision_recall_fscore_support(
    true_labels, preds, zero_division=0)

metrics_df = pd.DataFrame({
    'emotion': emotion_labels,
    'precision': precision,
    'recall': recall,
    'f1': f1s,
    'support': support
})

metrics_df.to_csv("emotion_classification_report.csv", index=False)
print("Per-label metrics saved to emotion_classification_report.csv")

# === Save F1 metrics ===
overall_metrics = {
    "macro_f1": f1_score(true_labels, preds, average="macro", zero_division=0),
    "micro_f1": f1_score(true_labels, preds, average="micro", zero_division=0),
    "samples_f1": f1_score(true_labels, preds, average="samples", zero_division=0)
}

with open("overall_metrics.json", "w") as f:
    json.dump(overall_metrics, f, indent=2)

print("Overall metrics saved to overall_metrics.json")

# === Step 10: Visualize the F1 score for each label (heatmap) ===
plt.figure(figsize=(12, 6))
sns.barplot(x='emotion', y='f1', data=metrics_df.sort_values('f1', ascending=False))
plt.xticks(rotation=90)
plt.title("F1 Scores per Emotion Label")
plt.ylabel("F1 Score")
plt.xlabel("Emotion")
plt.tight_layout()
plt.savefig("f1_scores_barplot.png")
plt.show()
print("F1 Heatmap saved as f1_scores_barplot.png")
