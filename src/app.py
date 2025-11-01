import data_utils as dataset
from fastapi import FastAPI
import mlflow
import os
from pydantic import BaseModel
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_MLFLOW_URI = "file:///app/mlruns"

RUN_ID = "827924430441517614"
SPECIFIC_ARTIFACT_PATH = "models/m-39b94320daed4fe894345d9e02c7c5a3/artifacts"
MODEL_ARTIFACT_PATH = f"file:///app/mlruns/{RUN_ID}/{SPECIFIC_ARTIFACT_PATH}"

tokenizer_uri = "src/models/final_tokenizer_assets"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_uri)
mlflow.set_tracking_uri(MODEL_MLFLOW_URI)
mlflow.set_registry_uri(MODEL_MLFLOW_URI)

PROJECT_ROOT_DIR = "/app"

if f"{PROJECT_ROOT_DIR}/src" not in sys.path:
    sys.path.append(f"{PROJECT_ROOT_DIR}/src")

sentiment_model = mlflow.pytorch.load_model(MODEL_ARTIFACT_PATH).to(device)

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(text: TextInput):
    cleaned_text = dataset.clean_data(text.text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    input_ids = inputs['input_ids']

    with torch.no_grad():
        outputs = sentiment_model(
            input_ids
        )
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs

        pred_class = torch.argmax(logits, dim=1).item()

    return {"text": text.text, "cleaned_text": cleaned_text, "prediction": int(pred_class)}

@app.get("/")
def home():
    return {"message": "Sentiment API is running"}