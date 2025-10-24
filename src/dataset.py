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

# Downloads
nltk.download('wordnet')
nltk.download('omw-1.4')

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