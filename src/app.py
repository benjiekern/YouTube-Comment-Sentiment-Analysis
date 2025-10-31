import dataset
from fastapi import FastAPI
import mlflow
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_uri = "file:///C:/Users/Benji/PycharmProjects/YouTube-Comment-Sentiment-Analysis/mlruns"
model_load_uri = "models:/YT_Sentiment_Model_LSTM/latest"
tokenizer_uri = "models/final_tokenizer_assets"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_uri)
mlflow.set_tracking_uri(model_uri)
mlflow.set_registry_uri(model_uri)
sentiment_model = mlflow.pytorch.load_model(model_load_uri).to(device)

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(text: TextInput):
    text = dataset.clean_data(text.text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = sentiment_model(**inputs)
        pred_class = torch.argmax(outputs.logits, dim=1).item()

    return {"text": text.text, "cleaned_text": text, "prediction": int(pred_class)}

@app.get("/")
def home():
    return {"message": "Sentiment API is running"}