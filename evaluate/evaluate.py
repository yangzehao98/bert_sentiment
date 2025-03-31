# evaluate.py

import torch
import numpy as np
import pandas as pd
import json
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, test_loader, device, threshold, label_names):
    """
    Evaluates the model on the test dataset and returns metrics.

    Args:
        model: Trained DistilBERT model.
        test_loader: DataLoader for test set.
        device: torch.device
        threshold: Probability threshold for classification.
        label_names: List of emotion labels.

    Returns:
        metrics_df: DataFrame containing per-label precision, recall, F1, and support.
        overall_metrics: Dictionary with macro/micro/samples averaged F1 scores.
    """
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
    preds = (all_probs >= threshold).astype(int)

    print("\nClassification Report:")
    print(classification_report(true_labels, preds, target_names=label_names, zero_division=0))

    precision, recall, f1s, support = precision_recall_fscore_support(
        true_labels, preds, zero_division=0
    )

    metrics_df = pd.DataFrame({
        'emotion': label_names,
        'precision': precision,
        'recall': recall,
        'f1': f1s,
        'support': support
    })

    # overall_metrics = {
    #     "macro_f1": f1_score(true_labels, preds, average="macro", zero_division=0),
    #     "micro_f1": f1_score(true_labels, preds, average="micro", zero_division=0),
    #     "samples_f1": f1_score(true_labels, preds, average="samples", zero_division=0)
    # }

    return metrics_df


def print_metrics(metrics_df):
    """
    Saves evaluation metrics to CSV and JSON, and generates F1 score plot.

    Args:
        metrics_df: DataFrame of per-label metrics.
    """
    # metrics_df.to_csv("emotion_classification_report.csv", index=False)
    # print("Per-label metrics saved to emotion_classification_report.csv")

    # with open("overall_metrics.json", "w") as f:
        # json.dump(overall_metrics, f, indent=2)
    # print("Overall metrics saved to overall_metrics.json")

    plt.figure(figsize=(12, 6))
    sns.barplot(x='emotion', y='f1', data=metrics_df.sort_values('f1', ascending=False))
    plt.xticks(rotation=90)
    plt.title("F1 Scores per Emotion Label")
    plt.ylabel("F1 Score")
    plt.xlabel("Emotion")
    plt.tight_layout()
    plt.savefig("f1_scores_barplot.png")
    plt.show()
    print("F1 score barplot saved to f1_scores_barplot.png")
