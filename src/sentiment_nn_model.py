


# In[36]:


# Evaluate Model
def evaluate_model(model, data_loader, criterion, device):
    model.eval() 
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['sequence'].to(device)
            labels = batch['label'].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predicted_classes = torch.max(outputs, 1)
            all_predictions.extend(predicted_classes.cpu().numpy())
            all_labels.extend(labels.squeeze().cpu().numpy())

    avg_loss = total_loss / len(data_loader.dataset)
    true_labels = np.array(all_labels)
    predicted_labels = np.array(all_predictions)
    val_accuracy = accuracy_score(true_labels, predicted_labels)
    model.train() 
    return avg_loss, val_accuracy


# In[37]:


import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

VOCAB_SIZE = 10000
EMBED_SIZE = 256
LSTM_UNITS = 256
BATCH_SIZE = 128
N_EPOCHS = 15 

# Sentiment Model
class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_size, lstm_units):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=lstm_units,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # Fully connected layers
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(lstm_units * 2, 64)
        self.relu = nn.ReLU()
        self.out = nn.Linear(64, 3)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Concatenate last hidden states from both directions
        h_forward = h_n[-2,:,:]
        h_backward = h_n[-1,:,:]
        h = torch.cat((h_forward, h_backward), dim=1)

        x = self.dropout(h)
        x = self.fc(x)
        x = self.relu(x)
        x = self.out(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = SentimentModel(VOCAB_SIZE, EMBED_SIZE, LSTM_UNITS)
class_weights = torch.tensor([1.0, 1.5, 1.0], device=device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=2
)

print("Model Architecture:\n", model)


# In[38]:


from torch.utils.data import Dataset

# Sentiment Dataset
class SentimentDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long).squeeze()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {
            'sequence': self.sequences[idx],
            'label': self.labels[idx]
        }
        return sample


# In[39]:


from torch.utils.data import DataLoader

# Pass our training and validation data through custom dataset
train_data = SentimentDataset(X_train, y_train)
val_data = SentimentDataset(X_test, y_test)

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


# In[40]:


from sklearn.metrics import accuracy_score
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Train model
def train_model(model, train_loader, val_loader, criterion, optimizer, n_epochs, device):
    model.to(device)

    best_val_loss = float('inf')
    patience = 2
    trigger_times = 0

    # Loop through epochs
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0

        # Loop through each batch in train_loader
        for i, batch in enumerate(train_loader):
            inputs = batch['sequence'].to(device)
            labels = batch['label'].to(device) 
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        # Evaluate train loss, val loss, and val accuracy
        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Print out model results
        print(f'Epoch {epoch+1}/{n_epochs}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Validation Accuracy: {val_accuracy:.4f}') 
        scheduler.step(val_loss)

train_model(model, train_loader, val_loader, criterion, optimizer, N_EPOCHS, device)


# In[ ]:




