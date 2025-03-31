from transformers import DistilBertForSequenceClassification


def load_model(num_labels: int):
    """
    Loads a DistilBERT model for multi-label classification.

    Args:
        num_labels (int): Number of emotion labels (classes).

    Returns:
        model: A DistilBertForSequenceClassification model instance.
    """
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )
    return model
