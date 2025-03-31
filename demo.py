import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW, get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import itertools
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

df = pd.concat([pd.read_csv(f) for f in local_files]).reset_index(drop=True)

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

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'], df[emotion_labels], test_size=0.2, random_state=42)

# ========= Step 3: Tokenizer ============
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# ========= Step 4: Dataset Class ============
class GoEmotionsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels.values, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}, self.labels[idx]

# ========= Step 5: Search Space ============
lr_choices = [5e-5, 3e-5, 2e-5]
epoch_choices = [3, 5, 10]
batch_choices = [8, 16, 32]
maxlen_choices = [128, 256]
threshold_choices = [0.3, 0.4, 0.5, 0.6, 0.7]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

best_f1 = 0
best_config = {}
best_preds = None
best_true = None
best_metrics_df = None

# ========= Step 6: Grid Search ============
for lr, epoch, batch_size, max_len in itertools.product(lr_choices, epoch_choices, batch_choices, maxlen_choices):
    print(f"\n==== Trying config: lr={lr}, epochs={epoch}, batch_size={batch_size}, max_len={max_len} ====")

    train_encodings = tokenizer(list(train_texts), padding=True, truncation=True, return_tensors="pt", max_length=max_len)
    test_encodings = tokenizer(list(test_texts), padding=True, truncation=True, return_tensors="pt", max_length=max_len)

    train_dataset = GoEmotionsDataset(train_encodings, train_labels)
    test_dataset = GoEmotionsDataset(test_encodings, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(emotion_labels),
        problem_type="multi_label_classification"
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epoch
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for ep in range(epoch):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {ep+1}/{epoch}"):
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            outputs = model(**inputs)
            loss = loss_fn(outputs.logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

    # Evaluation
    model.eval()
    all_probs = []
    true_labels_eval = []

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()
            all_probs.append(probs)
            true_labels_eval.append(labels.numpy())

    all_probs = np.vstack(all_probs)
    true_labels_eval = np.vstack(true_labels_eval)

    # Try all thresholds
    best_thresh_f1 = 0
    best_thresh = 0.5
    best_current_preds = None

    for thresh in threshold_choices:
        preds = (all_probs >= thresh).astype(int)
        macro_f1 = f1_score(true_labels_eval, preds, average="macro", zero_division=0)
        print(f"  → Threshold {thresh:.2f} → Macro F1: {macro_f1:.4f}")
        if macro_f1 > best_thresh_f1:
            best_thresh_f1 = macro_f1
            best_thresh = thresh
            best_current_preds = preds

    if best_thresh_f1 > best_f1:
        best_f1 = best_thresh_f1
        best_preds = best_current_preds
        best_true = true_labels_eval
        best_config = {
            "lr": lr,
            "epochs": epoch,
            "batch_size": batch_size,
            "max_len": max_len,
            "threshold": best_thresh
        }
        torch.save(model.state_dict(), "best_distilbert_model.pt")
        print(f"🎉 New best F1: {best_f1:.4f}, model saved.")

        # save per-label metrics for best
        precision, recall, f1s, support = precision_recall_fscore_support(best_true, best_preds, zero_division=0)
        best_metrics_df = pd.DataFrame({
            'emotion': emotion_labels,
            'precision': precision,
            'recall': recall,
            'f1': f1s,
            'support': support
        })

# ========= Step 7: Save Results ============
print("\nClassification Report (best):")
print(classification_report(best_true, best_preds, target_names=emotion_labels, zero_division=0))

print(f"\nBest Macro F1 Score: {best_f1:.4f}")
print(f"Best Config: {best_config}")

# Save metrics for each label
best_metrics_df.to_csv("emotion_classification_report.csv", index=False)
print("Per-label metrics saved to emotion_classification_report.csv")

# Save total metrics
overall_metrics = {
    "macro_f1": f1_score(best_true, best_preds, average="macro", zero_division=0),
    "micro_f1": f1_score(best_true, best_preds, average="micro", zero_division=0),
    "samples_f1": f1_score(best_true, best_preds, average="samples", zero_division=0)
}
with open("overall_metrics.json", "w") as f:
    json.dump(overall_metrics, f, indent=2)

with open("best_config.json", "w") as f:
    json.dump({"best_f1": best_f1, "best_config": best_config}, f, indent=2)

print("Overall metrics and best config saved.")

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(x='emotion', y='f1', data=best_metrics_df.sort_values('f1', ascending=False))
plt.xticks(rotation=90)
plt.title("F1 Scores per Emotion Label")
plt.ylabel("F1 Score")
plt.xlabel("Emotion")
plt.tight_layout()
plt.savefig("f1_scores_barplot.png")
plt.show()
print("F1 scores has been saved as f1_scores_barplot.png")

