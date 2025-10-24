import numpy as np
import torch
from sklearn.metrics import accuracy_score

# Evaluate Model
def evaluate_model(sentiment_model, data_loader, criterion, device):
    sentiment_model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            outputs = sentiment_model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predicted_classes = torch.max(outputs, 1)
            all_predictions.extend(predicted_classes.cpu().numpy())
            all_labels.extend(labels.squeeze().cpu().numpy())

    avg_loss = total_loss / len(data_loader.dataset)
    true_labels = np.array(all_labels)
    predicted_labels = np.array(all_predictions)
    val_accuracy = accuracy_score(true_labels, predicted_labels)
    sentiment_model.train()
    return avg_loss, val_accuracy