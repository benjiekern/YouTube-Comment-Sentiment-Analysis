# Imports
import nltk
from nltk.corpus import stopwords
from nltk.stem import *
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import warnings
import yaml

# Disable Warnings
warnings.filterwarnings('ignore')

# Load in config file
with open("../config.yaml") as f:
    config = yaml.safe_load(f)

# Downloads
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load in data from csv file
def load_data(config):
    df = pd.read_csv(config["data"]["path"])
    df.rename(columns={'Comment': 'comment', 'Sentiment': 'sentiment'}, inplace=True)
    df = df[df['sentiment'] != 'neutral']
    df.dropna(inplace=True)
    df = df[~df.duplicated()]
    return df

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Clean data
def clean_data(text):
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    tokens = text.split()
    processed = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    text = ' '.join(processed)
    return text.lower().strip()

# Preprocess data (clean data, encode labels, and split data into train and test sets)
def preprocess_data(df):
    df['comment'] = df['comment'].apply(clean_data)
    le = LabelEncoder()
    y = le.fit_transform(df['sentiment'])
    X_train, X_test, y_train, y_test = train_test_split(df['comment'], y, test_size=0.2, random_state=42)
    return X_train.tolist(), X_test.tolist(), y_train, y_test

# Dataset for text classification tasks
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': label
        }

df = load_data(config)
X_train, X_test, y_train, y_test = preprocess_data(df)

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

train_dataset = TextDataset(X_train, y_train, tokenizer, max_len=128)
test_dataset = TextDataset(X_test, y_test, tokenizer, max_len=128)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)