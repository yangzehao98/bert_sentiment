from data_engineering.config import EMOTION_LABELS, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, THRESHOLD
from data_engineering.data_loader import load_and_prepare_data, GoEmotionsDataset, get_tokenizer
from model.model import load_model
from model.train import train_model
from evaluate.evaluate import evaluate_model, print_metrics
from torch.utils.data import DataLoader
from transformers import get_scheduler
from torch.optim import AdamW
import torch


def main():
    print("\n[1] Loading and preprocessing data...")
    train_texts, test_texts, train_labels, test_labels = load_and_prepare_data()

    tokenizer = get_tokenizer()

    train_encodings = tokenizer(list(train_texts), padding=True, truncation=True, return_tensors="pt", max_length=128)
    test_encodings = tokenizer(list(test_texts), padding=True, truncation=True, return_tensors="pt", max_length=128)

    train_dataset = GoEmotionsDataset(train_encodings, train_labels)
    test_dataset = GoEmotionsDataset(test_encodings, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2 * BATCH_SIZE)

    print("\n[2] Initializing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(len(EMOTION_LABELS))
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader) * NUM_EPOCHS
    )

    print("\n[3] Starting training...")
    train_model(model, train_loader, optimizer, lr_scheduler, device, NUM_EPOCHS)

    print("\n[4] Evaluating model...")
    report_df = evaluate_model(model, test_loader, device, THRESHOLD, EMOTION_LABELS)

    print("\n[5] Printing metrics...")
    print_metrics(report_df)

    print("\nDone. Results saved to local files.")


if __name__ == "__main__":
    main()
