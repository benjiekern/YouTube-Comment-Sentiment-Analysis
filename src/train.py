import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer

import dataset
import evaluate
import mlflow
import mlflow.pytorch
import model
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
VOCAB_SIZE = tokenizer.vocab_size
N_EPOCHS = 15
EMBED_SIZE = 256
LSTM_UNITS = 256
BATCH_SIZE = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sentiment_model = model.SentimentModel(VOCAB_SIZE, EMBED_SIZE, LSTM_UNITS)
class_weights = torch.tensor([1.0, 1.5, 1.0], device=device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(sentiment_model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=2
)


def preprocess_data(df):
    df['comment'] = df['comment'].apply(dataset.clean_data)
    le = LabelEncoder()
    y = le.fit_transform(df['sentiment'])
    X_train, X_test, y_train, y_test = train_test_split(df['comment'], y, test_size=0.2, random_state=42)
    return X_train.tolist(), X_test.tolist(), y_train, y_test


# Load in data from csv file
def load_data(config):
    df = pd.read_csv(config["data"]["path"])
    df.rename(columns={'Comment': 'comment', 'Sentiment': 'sentiment'}, inplace=True)
    df = df[df['sentiment'] != 'neutral']
    df.dropna(inplace=True)
    df = df[~df.duplicated()]
    return df


# Load in config file
with open("../config.yaml") as f:
    config = yaml.safe_load(f)

df = load_data(config)
X_train, X_test, y_train, y_test = preprocess_data(df)

train_data = dataset.TextDataset(X_train, y_train, tokenizer, 128)
val_data = dataset.TextDataset(X_test, y_test, tokenizer, 128)

# Dataloaders for batching
train_loader = DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)
val_loader = DataLoader(
    dataset=val_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# Train model
def train_model(sentiment_model, train_loader, val_loader, criterion, optimizer, n_epochs, device):
    sentiment_model.to(device)
    best_val_accuracy = float('-inf')
    patience = 3
    trigger_times = 0

    # Loop through epochs
    for epoch in range(n_epochs):
        sentiment_model.train()
        running_loss = 0.0

        # Loop through each batch in train_loader
        for i, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            outputs = sentiment_model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * input_ids.size(0)

        # Evaluate train loss, val loss, and val accuracy
        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_accuracy = evaluate.evaluate_model(sentiment_model, val_loader, criterion, device)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            trigger_times = 0
            torch.save(sentiment_model.state_dict(), 'best_model.pt')
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Print out model results
        print(f'Epoch {epoch + 1}/{n_epochs}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Validation Accuracy: {val_accuracy:.4f}')
        scheduler.step(val_loss)

        mlflow.log_metric('train_loss', train_loss)
        mlflow.log_metric("val_loss", val_loss)
        mlflow.log_metric("val_accuracy", val_accuracy)

        scheduler.step(val_loss)

# print("Tracking URI:", mlflow.get_tracking_uri())
mlflow.set_tracking_uri("file:///C:/Users/Benji/PycharmProjects/YouTube-Comment-Sentiment-Analysis/mlruns")
print("Tracking URI:", mlflow.get_tracking_uri())

with mlflow.start_run():
    mlflow.log_param("embed_size", EMBED_SIZE)
    mlflow.log_param("lstm_units", LSTM_UNITS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("n_epochs", N_EPOCHS)
    mlflow.log_param("vocab_size", VOCAB_SIZE)

    example_input = torch.randint(0, VOCAB_SIZE, (1, 128))
    example_input = example_input.numpy().tolist()
    train_model(sentiment_model, train_loader, val_loader, criterion, optimizer, N_EPOCHS, device)

    mlflow.pytorch.log_model(sentiment_model, artifact_path="model", input_example=example_input)