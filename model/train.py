# train.py

import torch
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss

def train_model(model, train_loader, optimizer, scheduler, device, num_epochs):
    """
    Trains the DistilBERT multi-label classification model.

    Args:
        model: The DistilBERT model instance.
        train_loader: DataLoader for training data.
        optimizer: Optimizer (e.g., AdamW).
        scheduler: Learning rate scheduler.
        device: torch.device ('cuda' or 'cpu').
        num_epochs (int): Number of training epochs.
    """
    loss_fn = BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            outputs = model(**inputs)
            loss = loss_fn(outputs.logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
