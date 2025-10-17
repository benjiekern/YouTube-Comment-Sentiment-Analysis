# Imports
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import *
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
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

# Clean data
def clean_data(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    tokens = text.split()
    processed = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    text = ' '.join(processed)
    return text.lower().strip()

# Preprocess data, preparing it for model
def preprocess_data(df, config):
    df['comment'] = df['comment'].apply(clean_data)

    # Tokenize comments
    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    X = df['comment']
    tokenizer.fit_on_texts(X)
    X = tokenizer.texts_to_sequences(X)
    lengths = [len(sublist) for sublist in X]
    max_len = int(np.percentile(lengths, 90))
    X = pad_sequences(X, maxlen=max_len, padding='post', truncating='post')
    le = LabelEncoder()
    y = le.fit_transform(df['sentiment'])
    y = to_categorical(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)
    return X_train, X_test, y_train, y_test

reviews = load_data(config)
X_train, X_test, y_train, y_test = preprocess_data(reviews, config)